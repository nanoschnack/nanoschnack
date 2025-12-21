import asciichartpy
import time

def ascii_loss_plot(points, width=60, height=10):
    if len(points) < 2:
        return "loss (last hour): not enough data"
    t0, t1 = points[0][0], points[-1][0]
    if t1 == t0:
        return "loss (last hour): not enough data"
    bins = [[] for _ in range(width)]
    span = t1 - t0
    for t, loss in points:
        idx = int((t - t0) / span * (width - 1))
        bins[idx].append(loss)
    series = []
    last = None
    for b in bins:
        if b:
            val = sum(b) / len(b)
            last = val
            series.append(val)
        else:
            series.append(last)
    vals = [v for v in series if v is not None]
    if not vals:
        return "loss (last hour): not enough data"
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1e-6
    t_start = time.strftime("%H:%M:%S", time.localtime(t0))
    t_end = time.strftime("%H:%M:%S", time.localtime(t1))
    t_mid = time.strftime("%H:%M:%S", time.localtime((t0 + t1) / 2))
    header = f"loss (last hour, min {vmin:.4f} max {vmax:.4f})"
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
