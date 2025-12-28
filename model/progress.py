import random
import shutil
import time
import unicodedata
from collections import deque


class ProgressLogger:
    """Track loss history and emit periodic logs/plots for long-running training loops.

    Designed for streaming datasets where epoch length is unknown.
    Uses time-based intervals to control plotting cadence.
    Stores up to 3600 loss samples for the ASCII chart.
    """
    def __init__(
        self,
        plot_fn,
        start_global_step=0,
        start_total_samples=0,
        start_total_tokens=0,
        warmup_plot_interval=60,
        plot_interval=600,
        warmup_window_secs=600,
        estimated_total_tokens=None,
    ):
        # Keep configuration and bookkeeping for periodic logging.
        self.plot_fn = plot_fn
        self.global_step = start_global_step
        self.warmup_plot_interval = warmup_plot_interval
        self.plot_interval = plot_interval
        self.warmup_window_secs = warmup_window_secs
        self.start_time = time.time()
        self.last_tick_time = self.start_time
        self.last_plot_time = self.start_time
        self.total_samples = start_total_samples
        self.total_tokens = start_total_tokens
        self.estimated_total_tokens = estimated_total_tokens
        self.samples_per_sec = 0.0
        self.loss_history = deque()
        self.force_plot = False

    def tick(
        self,
        loss_value,
        batch_size,
        token_count,
        lr,
        epoch,
        step,
        loss_delta=None,
        remaining_tokens=None,
        io_time=None,
        gpu_time=None,
    ):
        # Record the latest loss and retain a rolling window for plotting.
        now = time.time()
        self.total_tokens += token_count
        self.loss_history.append((self.total_tokens, loss_value))
        if len(self.loss_history) > 3600:
            self.loss_history.popleft()

        # Log throughput and loss for every tick (caller controls cadence).
        self.total_samples += batch_size
        elapsed = now - self.last_tick_time
        samples_per_sec = batch_size / elapsed if elapsed > 0 else 0.0
        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0
        self.samples_per_sec = samples_per_sec
        if self.estimated_total_tokens:
            pct = (self.total_tokens / self.estimated_total_tokens) * 100
            eta = self._format_eta(
                remaining_tokens if remaining_tokens is not None else 0,
                tokens_per_sec,
            )
        else:
            pct = 0.0
            eta = "?"

        parts = [
            f"Tokens {self._format_count(self.total_tokens)}",
            f"Total {pct:.1f}%",
            f"Samples {self._format_count(self.total_samples)}",
            f"Epoch {epoch+1}",
            f"Steps {step+1}/{self.global_step+1}",
            f"Loss {self._format_loss(loss_value)}",
        ]
        if loss_delta is not None:
            parts[-1] = f"Loss {self._format_loss(loss_value)} (Δ{loss_delta:.2f})"
        if io_time is not None and gpu_time is not None:
            parts.append(f"IO {self._format_duration(io_time)}")
            parts.append(f"GPU {self._format_duration(gpu_time)}")
        parts.extend(
            [
                f"LR {self._format_lr(lr)}",
                f"Samples/s {self._format_rate(samples_per_sec)}",
                f"Tokens/s {self._format_rate(tokens_per_sec)}",
                f"ETA {eta}",
            ]
        )
        message = " | ".join(parts)
        print(message, flush=True)
        self.last_tick_time = now

        # Plot loss every minute for the first 10 minutes, then every 10 minutes.
        plot_printed = False

        # Honor explicit plot requests before checking time-based intervals.
        if self.force_plot:
            print(self.plot_fn(list(self.loss_history)))
            self.last_plot_time = now
            self.force_plot = False
            plot_printed = True
        else:
            interval = (
                self.warmup_plot_interval
                if (now - self.start_time) < self.warmup_window_secs
                else self.plot_interval
            )
            if now - self.last_plot_time >= interval:
                print(self.plot_fn(list(self.loss_history)))
                self.last_plot_time = now
                plot_printed = True

        # Keep a global step counter for resuming logs across restarts.
        self.global_step += 1
        return plot_printed

    def request_plot(self):
        # Allow callers to force a plot on the next tick.
        self.force_plot = True

    def print_dataset_pos(
        self,
        global_counts,
        resume_base,
        dataset_specs,
        total_rows_by_spec,
        avg_tokens_by_spec=None,
        est_tokens_by_spec=None,
        target_tokens,
    ):
        # Emit dataset position summaries for each spec.
        if avg_tokens_by_spec is None:
            avg_tokens_by_spec = {}
        if est_tokens_by_spec is None:
            est_tokens_by_spec = {}
        def _format_row_count(value):
            if value < 10000:
                return str(int(value))
            return self._format_compact(value)

        def _format_token_count(value):
            if value is None:
                return "?"
            return self._format_compact(int(value))

        print(
            f"Dataset Position: tokens={self.total_tokens} target={target_tokens}",
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
            if total_rows:
                pct = (current_rows / total_rows) * 100
                token_detail = (
                    f" tokens={_format_token_count(current_tokens)}"
                    f"/{_format_token_count(est_tokens)} ({tokens_pct:.1f}%)"
                    if tokens_pct is not None
                    else f" tokens={_format_token_count(current_tokens)}/{_format_token_count(est_tokens)}"
                )
                print(
                    f"  {spec_key}: resume={_format_row_count(resume_rows_count)} "
                    f"rows={_format_row_count(current_rows)}"
                    f"/{_format_row_count(total_rows)} ({pct:.1f}%)"
                    f"{token_detail}",
                    flush=True,
                )
            else:
                token_detail = (
                    f" tokens={_format_token_count(current_tokens)}"
                    f"/{_format_token_count(est_tokens)}"
                )
                print(
                    f"  {spec_key}: resume={_format_row_count(resume_rows_count)} "
                    f"rows={_format_row_count(current_rows)}"
                    f"{token_detail}",
                    flush=True,
                )

    def print_input_sample(
        self,
        rank,
        inputs,
        attention_mask,
        tokenizer,
        width=120,
        sample_index=None,
        source_ids=None,
        dataset_specs=None,
    ):
        # Emit a per-rank input sample for shard sanity checks.
        if sample_index is None:
            sample_index = random.randrange(inputs.size(0))
        input_ids = inputs[sample_index]
        if attention_mask is not None:
            mask = attention_mask[sample_index]
            if mask.size(0) == inputs.size(1) + 1:
                mask = mask[:-1]
            if mask.size(0) == inputs.size(1):
                input_ids = input_ids[mask.bool()]
        decoded_input = tokenizer.decode(input_ids.tolist())
        escaped = (
            decoded_input.replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        prefix = f"{rank} "
        if source_ids is not None and dataset_specs is not None:
            source_id = int(source_ids[sample_index])
            if 0 <= source_id < len(dataset_specs):
                prefix = f"{rank} {dataset_specs[source_id]['spec']}: "
        term_width = shutil.get_terminal_size((width, 20)).columns
        max_len = max(0, term_width - self._display_width(prefix))
        snippet = self._truncate_to_width(escaped, max_len)
        print(f"{prefix}{snippet}")

    def format_completion(self, prompt, completion, width=120):
        # Format a completion block with escaped, truncated content.
        escaped = (
            completion.replace("\\", "\\\\")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        )
        prefix = prompt
        term_width = shutil.get_terminal_size((width, 20)).columns
        max_len = max(0, term_width - self._display_width(prefix))
        snippet = self._truncate_to_width(escaped, max_len)
        return f"{prefix}{snippet}"

    def _format_eta(self, remaining_units, units_per_sec):
        # Format an ETA string from remaining samples and throughput.
        if units_per_sec <= 0:
            return "?"
        remaining_secs = remaining_units / units_per_sec
        hours = int(remaining_secs // 3600)
        minutes = int((remaining_secs % 3600) // 60)
        return f"{hours}h{minutes:02d}m"

    def _format_count(self, value):
        # Format counts with compact suffixes for readability.
        return self._format_compact(value)

    def _format_rate(self, value, width=6):
        # Format per-second rates with compact suffixes.
        text = self._format_compact(value, digits=4)
        return text.rjust(width) if len(text) < width else text

    def _format_compact(self, value, digits=3):
        # Keep compact count outputs consistent across totals and rates.
        if value >= 1_000_000_000:
            return f"{self._format_sig(value / 1_000_000_000, digits=digits)}b"
        if value >= 1_000_000:
            return f"{self._format_sig(value / 1_000_000, digits=digits)}m"
        if value >= 1_000:
            return f"{self._format_sig(value / 1_000, digits=digits)}k"
        return self._format_sig(value, digits=digits)

    def _format_sig(self, value, digits=3):
        # Format up to the requested significant digits for compact displays.
        if digits <= 3:
            if value >= 100:
                return f"{value:.0f}"
            if value >= 10:
                return f"{value:.1f}"
            return f"{value:.2f}"
        if value >= 1000:
            return f"{value:.0f}"
        if value >= 100:
            return f"{value:.1f}"
        if value >= 10:
            return f"{value:.2f}"
        return f"{value:.3f}"

    def _format_loss(self, value, width=5):
        # Cap loss to two digits before the decimal to stabilize width.
        text = f"{value:5.2f}" if value < 100 else f"{value:5.1f}"
        return text.rjust(width) if len(text) < width else text

    def _format_duration(self, seconds):
        if seconds < 1.0:
            return f"{seconds * 1000:3.0f}ms"
        if seconds < 10.0:
            return f"{seconds:.2f}s"
        return f"{seconds:.1f}s"

    def _format_lr(self, value, width=8):
        # Keep a fixed-width LR with six decimals.
        text = f"{value:.6f}"
        return text.rjust(width) if len(text) < width else text

    def _display_width(self, text):
        # Approximate terminal column width for escaped strings.
        width = 0
        for char in text:
            if unicodedata.combining(char):
                continue
            east_asian = unicodedata.east_asian_width(char)
            width += 2 if east_asian in ("W", "F") else 1
        return width

    def _truncate_to_width(self, text, max_width):
        # Truncate text to fit within the requested display width.
        if max_width <= 0:
            return ""
        width = 0
        out = []
        for char in text:
            if unicodedata.combining(char):
                out.append(char)
                continue
            east_asian = unicodedata.east_asian_width(char)
            char_width = 2 if east_asian in ("W", "F") else 1
            if width + char_width > max_width:
                break
            out.append(char)
            width += char_width
        return "".join(out)
