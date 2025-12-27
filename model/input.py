import select
import sys


def make_input_poller(is_master):
    """Return a non-blocking command poller for stdin.

    Uses select to avoid blocking the training loop.
    Only enabled for the master process when stdin is a TTY.
    """
    if not is_master or not sys.stdin.isatty():
        return lambda: None

    def _poll():
        # Non-blocking check for a command from stdin.
    # Returns None when no command is available.
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        line = sys.stdin.readline()
        return line.strip().lower()

    return _poll
