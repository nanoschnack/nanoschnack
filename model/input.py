import select
import sys


def make_plot_request_poller(is_master):
    """Return a non-blocking plot request poller for stdin.

    Uses select to avoid blocking the training loop.
    Only enabled for the master process when stdin is a TTY.
    """
    if not is_master or not sys.stdin.isatty():
        return lambda: False

    def _poll():
        # Non-blocking check for a plot request from stdin.
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return False
        line = sys.stdin.readline()
        return line.strip().lower() == "p"

    return _poll
