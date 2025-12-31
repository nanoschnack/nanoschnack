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
