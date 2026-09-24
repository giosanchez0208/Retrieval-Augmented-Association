"""Draw boxes and identities onto frames, styled like a paper figure.

The frame keeps its pixels; boxes are thin outlines with a small id tag, and a
light caption strip below the frame carries sequence, frame and time.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Categorical palette in fixed order (checked for colour-vision deficiencies).
# Ids cycle through it; every box also carries its id, so colour is never the
# only cue.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
HAIRLINE = "#e1e0d9"
NEUTRAL = "#898781"

_FONTS = ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc", "LiberationSans-Regular.ttf")


@lru_cache(maxsize=16)
def load_font(size: int) -> ImageFont.FreeTypeFont:
    for name in _FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def id_color(track_id: int) -> str:
    return PALETTE[(int(track_id) - 1) % len(PALETTE)]


def _rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


def _luminance(color: str) -> float:
    channels = [c / 255 for c in _rgb(color)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def text_color_on(fill: str) -> str:
    """Dark or white text, whichever contrasts more with ``fill``."""
    lum = _luminance(fill)
    return INK if (lum + 0.05) / (_luminance(INK) + 0.05) >= 1.05 / (lum + 0.05) else "#ffffff"


@dataclass(frozen=True)
class Box:
    xyxy: tuple[float, float, float, float]  # 0-based pixels in the source frame
    color: str
    label: str | None = None
    dashed: bool = False


class FigureRenderer:
    """Renders a frame plus boxes into a figure: the frame above a caption strip."""

    def __init__(self, width: int, height: int, scale: float = 1.0) -> None:
        self.scale = scale
        self.size = (round(width * scale), round(height * scale))
        unit = self.size[1] / 1080
        self.line_width = max(1, round(2 * unit))
        self.pad = max(2, round(4 * unit))
        self.tag_font = load_font(max(10, round(15 * unit)))
        self.caption_font = load_font(max(11, round(19 * unit)))
        self.caption_height = round(self.caption_font.size * 2.4)

    @property
    def figure_size(self) -> tuple[int, int]:
        return self.size[0], self.size[1] + self.caption_height

    def render(self, frame_bgr: np.ndarray, boxes: list[Box], caption: str, note: str = "") -> np.ndarray:
        image = Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        if image.size != self.size:
            image = image.resize(self.size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", self.figure_size, SURFACE)
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        for box in boxes:
            self._draw_box(draw, box)

        top = self.size[1]
        draw.line([(0, top), (self.size[0], top)], fill=HAIRLINE, width=1)
        middle = top + self.caption_height // 2
        margin = 3 * self.pad
        draw.text((margin, middle), caption, font=self.caption_font, fill=INK, anchor="lm")
        if note:
            draw.text((self.size[0] - margin, middle), note, font=self.caption_font, fill=INK_SECONDARY, anchor="rm")
        return np.asarray(canvas)[:, :, ::-1].copy()

    def _draw_box(self, draw: ImageDraw.ImageDraw, box: Box) -> None:
        w, h = self.size
        x1, y1, x2, y2 = (v * self.scale for v in box.xyxy)
        x1, x2 = max(x1, 0), min(x2, w - 1)
        y1, y2 = max(y1, 0), min(y2, h - 1)
        if x2 - x1 < 1 or y2 - y1 < 1:
            return
        if box.dashed:
            self._dashed_rectangle(draw, (x1, y1, x2, y2), box.color)
        else:
            draw.rectangle((x1, y1, x2, y2), outline=box.color, width=self.line_width)
        if box.label:
            left, top, right, bottom = draw.textbbox((0, 0), box.label, font=self.tag_font, anchor="lt")
            tag_w, tag_h = right - left + 2 * self.pad, bottom - top + 2 * self.pad
            tx = min(x1, w - tag_w)
            ty = y1 - tag_h if y1 - tag_h >= 0 else y1
            draw.rectangle((tx, ty, tx + tag_w, ty + tag_h), fill=box.color)
            draw.text(
                (tx + self.pad, ty + self.pad), box.label, font=self.tag_font,
                fill=text_color_on(box.color), anchor="lt",
            )

    def _dashed_rectangle(self, draw: ImageDraw.ImageDraw, xyxy: tuple[float, ...], color: str) -> None:
        x1, y1, x2, y2 = xyxy
        dash, gap = 4 * self.pad, 3 * self.pad
        for (ax, ay), (bx, by) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
            length = max(abs(bx - ax), abs(by - ay))
            pos = 0.0
            while pos < length:
                end = min(pos + dash, length)
                t0, t1 = pos / length, end / length
                draw.line(
                    [(ax + (bx - ax) * t0, ay + (by - ay) * t0), (ax + (bx - ax) * t1, ay + (by - ay) * t1)],
                    fill=color, width=self.line_width,
                )
                pos += dash + gap
