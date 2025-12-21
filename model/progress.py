import time
from collections import deque


class ProgressLogger:
    """Track loss history and emit periodic logs/plots for long-running training loops.

    Designed for streaming datasets where epoch length is unknown.
    Uses time-based intervals to control logging and plotting cadence.
    Stores up to 3600 loss samples for the ASCII chart.
    """
    def __init__(
        self,
        plot_fn,
        start_global_step=0,
        start_total_samples=0,
        start_total_tokens=0,
        enabled=True,
        log_interval=10,
        warmup_plot_interval=60,
        plot_interval=600,
        warmup_window_secs=600,
    ):
        # Keep configuration and bookkeeping for periodic logging.
        self.plot_fn = plot_fn
        self.global_step = start_global_step
        self.log_interval = log_interval
        self.warmup_plot_interval = warmup_plot_interval
        self.plot_interval = plot_interval
        self.warmup_window_secs = warmup_window_secs
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_plot_time = self.start_time
        self.has_logged = False
        self.samples_since_log = 0
        self.tokens_since_log = 0
        self.loss_since_log = 0.0
        self.loss_steps = 0
        self.total_samples = start_total_samples
        self.total_tokens = start_total_tokens
        self.samples_per_sec = 0.0
        self.enabled = enabled
        self.loss_history = deque()

    def tick(self, loss_value, batch_size, token_count, lr, epoch, step, shard_index, shard_count, shard_len, remaining_samples):
        # Record the latest loss and retain a rolling window for plotting.
        now = time.time()
        self.total_tokens += token_count
        self.loss_history.append((self.total_tokens, loss_value))
        if len(self.loss_history) > 3600:
            self.loss_history.popleft()

        # Log throughput and loss at the configured interval.
        self.samples_since_log += batch_size
        self.tokens_since_log += token_count
        self.loss_since_log += loss_value
        self.loss_steps += 1
        self.total_samples += batch_size
        if self.enabled and (not self.has_logged or (now - self.last_log_time >= self.log_interval)):
            elapsed = now - self.last_log_time
            avg_loss = self.loss_since_log / self.loss_steps if self.loss_steps else loss_value
            samples_per_sec = self.samples_since_log / elapsed if elapsed > 0 else 0.0
            tokens_per_sec = self.tokens_since_log / elapsed if elapsed > 0 else 0.0
            self.samples_per_sec = samples_per_sec
            estimated_total = shard_len * shard_count
            pct = min(100.0, (self.total_samples / estimated_total) * 100)
            shard_label = f"Shard {shard_index + 1}/{shard_count}"
            total_label = f"Total {pct:.1f}%"
            prefix = f"{shard_label}, {total_label}"

            message = (
                f"Tokens {self._format_count(self.total_tokens)}, "
                f"Total {pct:.1f}%, "
                f"Samples {self._format_count(self.total_samples)}, "
                f"Epoch {epoch+1}, "
                f"Step {step+1}, "
                f"Global {self.global_step+1}, "
                f"Shard {shard_index + 1}/{shard_count}, "
                f"Loss {avg_loss:.4f}, "
                f"LR {lr:.2e}, "
                f"Samples/s {samples_per_sec:.1f}, "
                f"Tokens/s {tokens_per_sec:.1f}, "
                f"ETA {self._format_eta(remaining_samples, samples_per_sec)}"
            )
            print(message, flush=True)
            self.last_log_time = now
            self.has_logged = True
            self.samples_since_log = 0
            self.tokens_since_log = 0
            self.loss_since_log = 0.0
            self.loss_steps = 0

        # Plot loss every minute for the first 10 minutes, then every 10 minutes.
        interval = (
            self.warmup_plot_interval
            if (now - self.start_time) < self.warmup_window_secs
            else self.plot_interval
        )
        if self.enabled and (now - self.last_plot_time >= interval):
            print(self.plot_fn(list(self.loss_history)))
            self.last_plot_time = now

        # Keep a global step counter for resuming logs across restarts.
        self.global_step += 1

    def _format_eta(self, remaining_samples, samples_per_sec):
        # Format an ETA string from remaining samples and throughput.
        if samples_per_sec <= 0:
            return "?"
        remaining_secs = remaining_samples / samples_per_sec
        hours = int(remaining_secs // 3600)
        minutes = int((remaining_secs % 3600) // 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m"
        return f"{minutes}m"

    def _format_count(self, value):
        # Format counts with compact suffixes for readability.
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}m"
        if value >= 1_000:
            return f"{value / 1_000:.1f}k"
        return str(int(value))
