import random
import shutil
import time

from text_format import display_width, truncate_to_width


class ProgressLogger:
    """Track progress and emit periodic logs for long-running training loops.

    Designed for streaming datasets where epoch length is unknown.
    Stores running counters for tokens, samples, and steps.
    Keeps throughput estimates for logging and ETA display.
    """
    def __init__(
        self,
        start_global_step=0,
        start_total_samples=0,
        start_total_tokens=0,
        estimated_total_tokens=None,
    ):
        # Keep configuration and bookkeeping for periodic logging.
        self.global_step = start_global_step
        self.start_time = time.time()
        self.last_tick_time = self.start_time
        self.total_samples = start_total_samples
        self.total_tokens = start_total_tokens
        self.estimated_total_tokens = estimated_total_tokens
        self.samples_per_sec = 0.0

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
        io_time=0.0,
        gpu_time=0.0,
        sync_time=0.0,
    ):
        # Log throughput and loss for every tick (caller controls cadence).
        now = time.time()
        self.total_tokens += token_count
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

        step_label = f"{step+1}"
        if step != self.global_step:
            step_label = f"{step+1}/{self.global_step+1}"
        parts = [
            step_label,
            f"Tokens {self._format_count(self.total_tokens)}",
            f"Total {pct:.1f}%",
            f"Samples {self._format_count(self.total_samples)}",
            f"Epoch {epoch+1}",
            f"Loss {self._format_loss(loss_value)}",
        ]
        if loss_delta is not None:
            parts[-1] = f"Loss {self._format_loss(loss_value)} (Δ{loss_delta:.2f})"
        parts.append(
            "Wait IO/GPU/Sync "
            f"{self._format_duration(io_time)}/"
            f"{self._format_duration(gpu_time)}/"
            f"{self._format_duration(sync_time)}"
        )
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

        # Keep a global step counter for resuming logs across restarts.
        self.global_step += 1

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
        max_len = max(0, term_width - display_width(prefix))
        snippet = truncate_to_width(escaped, max_len)
        print(f"{prefix}{snippet}")

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
        # Keep a fixed-width duration for log alignment (5 chars).
        if seconds < 1.0:
            ms = int(seconds * 1000)
            return f"{ms:>3d}ms"
        if seconds < 100.0:
            return f"{seconds:>4.1f}s"
        return f"{int(seconds):>4d}s"

    def _format_lr(self, value, width=8):
        # Keep a fixed-width LR with six decimals.
        text = f"{value:.6f}"
        return text.rjust(width) if len(text) < width else text
