import time
from collections import deque
from dataclasses import dataclass, field

import asciichartpy
import torch
import torch.distributed as dist

from chat import generate_reply_stream
from ddp_debug import log_ddp_debug
from text_format import format_compact, format_completion
from tokenizer import DATASET_EOS_TOKEN


@dataclass
class Plotter:
    """Manage loss plotting for training without progress coupling.
    Tracks loss history and token counts for charting.
    Handles plot timing with warmup and steady-state intervals.
    Supports on-demand plot requests from the training loop.
    """
    plot_fn: callable
    warmup_plot_interval: float
    plot_interval: float
    warmup_window_secs: float
    start_time: float = field(default_factory=time.time)
    last_plot_time: float = field(default_factory=time.time)
    total_tokens: int = 0
    loss_history: deque = field(default_factory=deque)
    plot_due: bool = False

    def record(self, token_count, loss_value):
        self.total_tokens += token_count
        self.loss_history.append((self.total_tokens, loss_value))
        if len(self.loss_history) > 3600:
            self.loss_history.popleft()

    def tick(
        self,
        token_count,
        loss_value,
        ddp_enabled=False,
        is_master=False,
        device=None,
        ddp_world_size=None,
        debug_payload=None,
    ):
        self.record(token_count, loss_value)
        plot_printed = False
        if is_master:
            now = time.time()
            if self.should_plot(now):
                self.print_plot(now)
                plot_printed = True

        if ddp_enabled:
            plot_flag = torch.tensor(1 if (is_master and plot_printed) else 0, device=device)
            dist.broadcast(plot_flag, src=0)
            plot_printed = bool(plot_flag.item())
            if plot_printed and debug_payload is not None:
                micro_loss, micro_tokens, micro_samples = debug_payload
                log_ddp_debug(
                    ddp_world_size,
                    micro_loss,
                    micro_tokens,
                    micro_samples,
                    device,
                    is_master,
                )

        return plot_printed

    def request_plot(self):
        self.plot_due = True

    def should_plot(self, now):
        if self.plot_due:
            return True
        interval = (
            self.warmup_plot_interval
            if (now - self.start_time) < self.warmup_window_secs
            else self.plot_interval
        )
        return now - self.last_plot_time >= interval

    def print_plot(self, now):
        print(self.plot_fn(list(self.loss_history)))
        self.last_plot_time = now
        self.plot_due = False

    def print_dataset_pos(
        self,
        total_tokens,
        global_counts,
        resume_base,
        dataset_specs,
        total_rows_by_spec,
        target_tokens,
        avg_tokens_by_spec=None,
        est_tokens_by_spec=None,
    ):
        # Emit dataset position summaries for each spec.
        if avg_tokens_by_spec is None:
            avg_tokens_by_spec = {}
        if est_tokens_by_spec is None:
            est_tokens_by_spec = {}
        def _format_row_count(value):
            if value < 10000:
                return str(int(value))
            return format_compact(value)

        def _format_token_count(value):
            if value is None:
                return "?"
            return format_compact(int(value))

        print(
            f"Dataset Position: tokens={total_tokens} target={target_tokens}",
            flush=True,
        )
        for spec in dataset_specs:
            spec_key = spec["spec"]
            current_rows = global_counts.get(spec_key, 0) + resume_base.get(spec_key, 0)
            resume_rows_count = resume_base.get(spec_key, 0)
            total_rows = total_rows_by_spec.get(spec_key)
            avg_tokens = avg_tokens_by_spec.get(spec_key)
            est_tokens = est_tokens_by_spec.get(spec_key)
            current_tokens = None
            if avg_tokens is not None:
                current_tokens = int(current_rows * avg_tokens)
            tokens_pct = None
            if est_tokens:
                tokens_pct = (current_tokens or 0) / est_tokens * 100
            pct = None
            if total_rows:
                pct = (current_rows / total_rows) * 100
            if pct is None and tokens_pct is not None:
                pct = tokens_pct
            pct_label = f" ({pct:.1f}%)" if pct is not None else ""
            token_detail = (
                f" tokens={_format_token_count(current_tokens)}"
                f"/{_format_token_count(est_tokens)}"
            )
            if total_rows:
                print(
                    f"  {spec_key}: resume={_format_row_count(resume_rows_count)} "
                    f"rows={_format_row_count(current_rows)}"
                    f"/{_format_row_count(total_rows)}"
                    f"{token_detail}{pct_label}",
                    flush=True,
                )
            else:
                print(
                    f"  {spec_key}: resume={_format_row_count(resume_rows_count)} "
                    f"rows={_format_row_count(current_rows)}"
                    f"{token_detail}{pct_label}",
                    flush=True,
                )


def plot_with_completion(points, model, tokenizer, config, device):
    """Render a loss chart with a sample completion appended."""
    # Render the loss plot first so completion failures don't block logs.
    chart = ascii_loss_plot(points)

    # Append the configured completion snapshot.
    was_training = model.training
    if was_training:
        model.eval()
    try:
        completion_prompt = f"{DATASET_EOS_TOKEN}{config.PLOT_COMPLETION_PROMPT}"
        reply_parts = []
        for token in generate_reply_stream(
                model,
                tokenizer,
                completion_prompt,
                context_len=config.CONTEXT_LEN,
                max_new_tokens=config.PLOT_COMPLETION_TOKENS,
                temperature=config.TEMPERATURE,
                top_k=config.TOP_K,
                device=device,
        ):
            reply_parts.append(token)
        completion = "".join(reply_parts)
    except Exception as exc:
        completion = f" [generation failed: {exc}]"
    finally:
        if was_training:
            model.train()
    formatted = format_completion(
        "Validation: ",
        f"{config.PLOT_COMPLETION_PROMPT}|>{completion}",
    )
    return f"{chart}\n{formatted}\n"


def ascii_loss_plot(points, width=60, height=10):
    """Render a compact ASCII loss chart for the most recent samples.

    Compresses tokens into a fixed-width chart, with start/mid/end labels.
    Accepts a list of (token_count, loss) points and returns a formatted string.
    """
    # Ensure we have enough data to produce a meaningful chart.
    if len(points) < 2:
        return "loss (tokens): not enough data"
    t0, t1 = points[0][0], points[-1][0]
    if t1 == t0:
        return "loss (tokens): not enough data"

    # Bucket losses into a fixed-width series for plotting.
    bins = [[] for _ in range(width)]
    span = t1 - t0
    for t, loss in points:
        idx = int((t - t0) / span * (width - 1))
        bins[idx].append(loss)

    # Fill gaps with the last known value to keep the chart continuous.
    series = []
    last = None
    for b in bins:
        if b:
            val = sum(b) / len(b)
            last = val
            series.append(val)
        else:
            series.append(last)

    # Normalize the series bounds for the header and chart.
    vals = [v for v in series if v is not None]
    if not vals:
        return "loss (tokens): not enough data"
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1e-6

    # Render the chart and align token counts under the plot area.
    def format_tokens(value):
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return str(int(value))

    t_start = format_tokens(t0)
    t_end = format_tokens(t1)
    t_mid = format_tokens((t0 + t1) / 2)
    header = f"loss (tokens, min {vmin:.4f} max {vmax:.4f})"
    series = [vmin if v is None else v for v in series]
    chart = asciichartpy.plot(series, {"height": height})
    chart_lines = chart.splitlines()
    content_width = max((len(line) for line in chart_lines), default=width)
    axis_chars = {"┤", "┼", "┬", "┴", "┌", "┐", "└", "┘"}
    plot_start = None
    for line in chart_lines:
        for idx, ch in enumerate(line):
            if ch in axis_chars:
                plot_start = idx + 1
                break
        if plot_start is not None:
            break
    if plot_start is None:
        plot_start = 0
    if width < 10:
        time_axis = f"{t_start} - {t_end}"
    else:
        left = t_start
        right = t_end
        mid = t_mid
        line = [" "] * content_width
        line[plot_start:plot_start + len(left)] = list(left)
        mid_start = max(plot_start + (content_width - plot_start - len(mid)) // 2, plot_start + len(left) + 1)
        line[mid_start:mid_start + len(mid)] = list(mid)
        right_start = max(content_width - len(right), mid_start + len(mid) + 1)
        line[right_start:right_start + len(right)] = list(right)
        time_axis = "".join(line).rstrip()
    return header + "\n" + chart + "\n" + time_axis
