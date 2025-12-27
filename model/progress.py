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

    def tick(
        self,
        loss_value,
        batch_size,
        token_count,
        lr,
        epoch,
        step,
        remaining_tokens=None,
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

        message = (
            f"Tokens {self._format_count(self.total_tokens)} | "
            f"Total {pct:.1f}% | "
            f"Samples {self._format_count(self.total_samples)} | "
            f"Epoch {epoch+1} | "
            f"Step {step+1} | "
            f"Global {self.global_step+1} | "
            f"Loss {self._format_loss(loss_value)} | "
            f"LR {self._format_lr(lr)} | "
            f"Samples/s {self._format_rate(samples_per_sec)} | "
            f"Tokens/s {self._format_rate(tokens_per_sec)} | "
            f"ETA {eta}"
        )
        print(message, flush=True)
        self.last_tick_time = now

        # Plot loss every minute for the first 10 minutes, then every 10 minutes.
        plot_printed = False
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

    def print_input_sample(self, rank, inputs, attention_mask, tokenizer, width=120, sample_index=None):
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
        prefix = f"{rank}: "
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
        return f"{hours:>4d}h{minutes:02d}m"

    def _format_count(self, value, width=5):
        # Format counts with fixed-width compact suffixes for readability.
        text = self._format_compact(value)
        return text.rjust(width) if len(text) < width else text

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

    def _format_loss(self, value, width=6):
        # Cap loss to two digits before the decimal to stabilize width.
        text = f"{value:5.2f}" if value < 100 else f"{value:5.1f}"
        return text.rjust(width) if len(text) < width else text

    def _format_lr(self, value, width=9):
        # Allow non-exponent LR while keeping a fixed width.
        text = f"{value:.8f}".rstrip("0").rstrip(".")
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
