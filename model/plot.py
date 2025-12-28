import asciichartpy

from chat import generate_reply_stream


def plot_with_completion(points, model, tokenizer, config, device, progress):
    """Render a loss chart with a sample completion appended."""
    # Render the loss plot first so completion failures don't block logs.
    chart = ascii_loss_plot(points)

    # Append the configured completion snapshot.
    was_training = model.training
    if was_training:
        model.eval()
    try:
        reply_parts = []
        for token in generate_reply_stream(
                model,
                tokenizer,
                config.PLOT_COMPLETION_PROMPT,
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
    formatted = progress.format_completion(
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
