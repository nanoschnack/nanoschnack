import shutil
import unicodedata


def display_width(text):
    # Approximate terminal column width for escaped strings.
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        east_asian = unicodedata.east_asian_width(char)
        width += 2 if east_asian in ("W", "F") else 1
    return width


def truncate_to_width(text, max_width):
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


def format_completion(prompt, completion, width=120):
    # Format a completion block with escaped, truncated content.
    escaped = (
        completion.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    prefix = prompt
    term_width = shutil.get_terminal_size((width, 20)).columns
    max_len = max(0, term_width - display_width(prefix))
    snippet = truncate_to_width(escaped, max_len)
    return f"{prefix}{snippet}"


def format_sig(value, digits=3):
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


def format_compact(value, digits=3):
    # Keep compact count outputs consistent across totals and rates.
    if value >= 1_000_000_000:
        return f"{format_sig(value / 1_000_000_000, digits=digits)}b"
    if value >= 1_000_000:
        return f"{format_sig(value / 1_000_000, digits=digits)}m"
    if value >= 1_000:
        return f"{format_sig(value / 1_000, digits=digits)}k"
    return format_sig(value, digits=digits)
