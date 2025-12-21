import time
from collections import deque


class ProgressLogger:
    """Track loss history and emit periodic logs/plots for long-running training loops.

    Designed for streaming datasets where epoch length is unknown.
    Uses time-based intervals to control logging and plotting cadence.
    Stores up to one hour of losses for the ASCII chart.
    """
    def __init__(
        self,
        plot_fn,
        start_global_step=0,
        start_total_samples=0,
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
        self.total_samples = start_total_samples
        self.loss_history = deque()

    def tick(self, loss_value, batch_size, epoch, step):
        # Record the latest loss and retain a rolling one-hour window.
        now = time.time()
        self.loss_history.append((now, loss_value))
        while self.loss_history and (now - self.loss_history[0][0]) > 3600:
            self.loss_history.popleft()

        # Log throughput and loss at the configured interval.
        self.samples_since_log += batch_size
        self.total_samples += batch_size
        if not self.has_logged or (now - self.last_log_time >= self.log_interval):
            elapsed = now - self.last_log_time
            samples_per_sec = self.samples_since_log / elapsed if elapsed > 0 else 0.0
            print(
                f"Epoch {epoch+1} (Step {step+1}, Global {self.global_step+1}), "
                f"Samples {self.total_samples:,}, "
                f"Loss: {loss_value:.4f}, Samples/s: {samples_per_sec:.1f}",
                flush=True,
            )
            self.last_log_time = now
            self.has_logged = True
            self.samples_since_log = 0

        # Plot loss every minute for the first 10 minutes, then every 10 minutes.
        interval = (
            self.warmup_plot_interval
            if (now - self.start_time) < self.warmup_window_secs
            else self.plot_interval
        )
        if now - self.last_plot_time >= interval:
            print(self.plot_fn(list(self.loss_history)))
            self.last_plot_time = now

        # Keep a global step counter for resuming logs across restarts.
        self.global_step += 1
