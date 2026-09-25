"""Draw the README's figures as SVG, with the standard library only.

    python docs/figures/make_figures.py

The numbers are the measured results quoted in the README, each with the command that
produced it. Style: white card, thin lines, small labels, Okabe-Ito colors.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent
INK, MUTED, GRID = "#222222", "#777777", "#e6e6e6"
MINE, DEEPSORT, LIGHT, BLOCK, GRAY = "#0072B2", "#E69F00", "#56B4E9", "#D55E00", "#999999"
FONT = "Helvetica, Arial, sans-serif"


def _svg(width: int, height: int, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'font-family="{FONT}">\n<rect width="100%" height="100%" fill="#ffffff"/>\n' + "\n".join(body) + "\n</svg>\n")


def _text(x, y, s, size=11, anchor="start", color=INK, weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" '
            f'font-weight="{weight}">{s}</text>')


def _line(x1, y1, x2, y2, color=GRID, width=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>'


def _rect(x, y, w, h, color):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" fill="{color}"/>'


def _circle(x, y, r, color, stroke=None):
    s = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"{s}/>'


def _legend(items, x, y):
    out = []
    for label, color in items:
        out += [_rect(x, y - 8, 10, 10, color), _text(x + 14, y, label, 10, color=MUTED)]
        x += 20 + 6.2 * len(label)
    return out


def hbars(path, title, labels, series, xmax, ticks, value_fmt="{:.1f}", ref=None, label_w=170, plot_w=360, note=None):
    """Horizontal bars, one group per label, one bar per series (name, values, color or per-bar colors)."""
    bar_h, gap = 11, 9
    group_h = bar_h * len(series) + gap
    top = 46 if len(series) > 1 else 34
    height = top + group_h * len(labels) + 34 + (14 if note else 0)
    width = label_w + plot_w + 60
    scale = plot_w / xmax
    body = [_text(12, 20, title, 13, weight="bold")]
    if len(series) > 1:
        body += _legend([(n, c if isinstance(c, str) else c[0]) for n, _, c in series], label_w, 38)
    axis_y = top + group_h * len(labels)
    for t in ticks:
        x = label_w + t * scale
        body += [_line(x, top - 4, x, axis_y), _text(x, axis_y + 14, f"{t:g}", 10, "middle", MUTED)]
    if ref is not None:
        x = label_w + ref[0] * scale
        body += [_line(x, top - 6, x, axis_y, MUTED, 1, "3,3"), _text(x + 3, top - 8, ref[1], 10, color=MUTED)]
    for i, label in enumerate(labels):
        y0 = top + i * group_h
        body.append(_text(label_w - 8, y0 + group_h / 2, label, 11, "end"))
        for k, (_, values, color) in enumerate(series):
            v = values[i]
            if v is None:
                continue
            c = color if isinstance(color, str) else color[i]
            y = y0 + k * bar_h
            body += [_rect(label_w, y, v * scale, bar_h - 2, c),
                     _text(label_w + v * scale + 4, y + bar_h - 3, value_fmt.format(v), 10, color=MUTED)]
    body.append(_line(label_w, top - 4, label_w, axis_y, MUTED))
    if note:
        body.append(_text(12, height - 10, note, 10, color=MUTED))
    (OUT / path).write_text(_svg(width, height, body), encoding="utf-8")


def stress_test():
    # python -m reidtrack.retrieval.probe --models osnet_x0_5_mot17
    probes = ["clean", "dark (x0.5)", "overexposed (x1.6)", "low contrast (glare)", "warm cast", "cool cast",
              "hard shadow over one side", "blocked below (40%)", "blocked above (40%)", "blocked side (35%)",
              "blocked anywhere (25-40%)"]
    values = STRESS["osnet_x0_5_mot17"]
    colors = [GRAY] + [LIGHT] * 6 + [BLOCK] * 4
    hbars("stress_test.svg", "Recognizing unseen people when the query sighting changes (mAP)", probes,
          [("OSNet x0.5", values, colors)], 90, [0, 20, 40, 60, 80], ref=(values[0], f"clean {values[0]:.1f}"),
          note="Blue: lighting changes. Red: part of the person covered by a piece of someone else. Gallery unchanged.")


def idsw_anatomy():
    # idsw_anatomy on raa_pairwise_x0_5_iou_last_val and hyst_s{1,2,3}_0 (mean), and on deepsort_osnet_x0_5_mot17_calibrated
    labels = ["swap: took someone else's ID", "flip back a moment later", "lost, then restarted", "other"]
    hbars("idsw_anatomy.svg", "Why IDs changed on the validation half", labels,
          [("mine, mean of 4 runs", [40.0, 28.0, 28.5, 3.0], MINE), ("DeepSORT", [36, 19, 41, 6], DEEPSORT)],
          50, [0, 10, 20, 30, 40, 50], value_fmt="{:g}", label_w=210)


def restarts():
    # forget_diag over the four final matchers
    hbars("restarts.svg", "\"Lost, then restarted\" switches: why the person got a new ID", [
        "still in the bank, not recognized", "forgotten after walking out (0.5 s)", "forgotten while hidden (5 s)"],
        [("mean of 4 runs", [20.75, 4.5, 3.25], [MINE, GRAY, GRAY])], 25, [0, 5, 10, 15, 20, 25], value_fmt="{:g}",
        label_w=245)


def frame_skip():
    # python -m reidtrack --stride {1,2,3} (4 to 6 trained matchers) and python -m reidtrack.baselines --stride ...
    panels = [("HOTA", [52.58, 50.95, 49.70], [51.9, 50.0, 48.3], (47, 54)),
              ("IDF1", [61.72, 60.58, 59.38], [60.6, 59.2, 57.2], (56, 63))]
    pw, ph, left, top = 230, 150, 50, 70
    width, height = left + 2 * (pw + 70), top + ph + 56
    body = [_text(12, 20, "Tracking every n-th frame, as an edge device would", 13, weight="bold")]
    body += _legend([("mine", MINE), ("DeepSORT", DEEPSORT)], left, 38)
    for p, (name, mine, deep, (lo, hi)) in enumerate(panels):
        x0 = left + p * (pw + 70)
        def px(i):
            return x0 + 20 + i * (pw - 40) / 2
        def py(v):
            return top + ph - (v - lo) / (hi - lo) * ph
        for t in range(lo, hi + 1):
            body += [_line(x0, py(t), x0 + pw, py(t)), _text(x0 - 6, py(t) + 3, f"{t}", 10, "end", MUTED)]
        for i, lab in enumerate(["every frame", "every 2nd", "every 3rd"]):
            body.append(_text(px(i), top + ph + 16, lab, 10, "middle", MUTED))
        for values, color in ((mine, MINE), (deep, DEEPSORT)):
            pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
            body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            for i, v in enumerate(values):
                body += [_circle(px(i), py(v), 3, color), _text(px(i) + 6, py(v) - 5, f"{v:.1f}", 10, color=color)]
        body.append(_text(x0, top - 10, name, 11, weight="bold"))
    body.append(_text(12, height - 10, "Skipped frames are filled in by interpolation for scoring. Mine: 4 to 6 trained "
                                       "matchers per point.", 10, color=MUTED))
    (OUT / "frame_skip.svg").write_text(_svg(width, height, body), encoding="utf-8")


def idsw_noise():
    # every trained run of each configuration on the validation half; DeepSORT is deterministic at 102
    rows = [
        ("learned, 21 cues", [108, 107, 106]),
        ("learned, 22 cues (final)", [108, 97, 97, 96, 95, 97]),
        ("+ camera compensation", [97, 99, 99, 110, 90, 96]),
        ("+ competition margins", [108, 113, 116]),
        ("+ on-policy rounds", [105, 110, 113]),
        ("+ hysteresis 0.05", [105, 100, 104, 93]),
        ("seed ensembles of 5", [103, 96]),
    ]
    lo, hi, label_w, pw, row_h, top = 85, 125, 170, 360, 22, 52
    width, height = label_w + pw + 40, top + row_h * len(rows) + 46
    def px(v):
        return label_w + (v - lo) / (hi - lo) * pw
    body = [_text(12, 20, "ID switches per trained run, validation half", 13, weight="bold")]
    axis_y = top + row_h * len(rows)
    for t in range(lo, hi + 1, 5):
        body += [_line(px(t), top - 4, px(t), axis_y), _text(px(t), axis_y + 14, f"{t}", 10, "middle", MUTED)]
    body += [_line(px(102), top - 8, px(102), axis_y, DEEPSORT, 1.5, "4,3"), _text(px(102) + 4, top - 10, "DeepSORT 102", 10, color=DEEPSORT)]
    for i, (label, runs) in enumerate(rows):
        y = top + i * row_h + row_h / 2
        body.append(_text(label_w - 8, y + 4, label, 11, "end", weight="bold" if "final" in label else "normal"))
        for v in runs:
            body.append(_circle(px(v), y, 4, MINE if "final" in label else GRAY, "#ffffff"))
        mean = sum(runs) / len(runs)
        body.append(_line(px(mean), y - 7, px(mean), y + 7, INK, 1.5))
    body.append(_text(12, height - 22, "Dots: single trained runs. Black tick: mean. Attention (149 to 379) is off the chart.",
                      10, color=MUTED))
    body.append(_text(12, height - 8, "Single runs swing by up to 20, as much as any effect tried.", 10, color=MUTED))
    (OUT / "idsw_noise.svg").write_text(_svg(width, height, body), encoding="utf-8")


STRESS = {  # python -m reidtrack.retrieval.probe; order as in stress_test()
    "osnet_x0_5_mot17": [78.1, 76.1, 74.3, 68.0, 76.8, 76.4, 73.1, 59.4, 32.7, 68.3, 52.0],
}

if __name__ == "__main__":
    for draw in (idsw_anatomy, restarts, frame_skip, idsw_noise):
        draw()
    if all(v is not None for v in STRESS.values()):
        stress_test()
