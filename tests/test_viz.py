from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from reidtrack.viz import PALETTE, Box, FigureRenderer, id_color, text_color_on
from reidtrack.viz import __main__ as viz_cli
from reidtrack.viz.render import SURFACE


def _bgr(hex_color):
    return [int(hex_color[i : i + 2], 16) for i in (5, 3, 1)]


def test_ids_take_palette_colours_in_fixed_order():
    assert id_color(1) == PALETTE[0]
    assert id_color(8) == PALETTE[7]
    assert id_color(9) == PALETTE[0]


def test_tag_text_contrasts_with_its_fill():
    assert text_color_on("#eda100") == "#0b0b0b"  # yellow: dark text
    assert text_color_on("#4a3aa7") == "#ffffff"  # violet: white text


def test_figure_is_the_frame_above_a_light_caption_strip():
    renderer = FigureRenderer(192, 108)
    frame = np.zeros((108, 192, 3), np.uint8)

    figure = renderer.render(frame, [Box((40, 50, 80, 100), PALETTE[0], "3")], "MOT17-02 · frame 1", note="ground truth")

    assert figure.shape == (108 + renderer.caption_height, 192, 3)
    assert figure[100, 60].tolist() == _bgr(PALETTE[0])  # bottom edge of the box
    assert figure[-2, -2].tolist() == _bgr(SURFACE)


def test_scaling_shrinks_frame_and_boxes():
    renderer = FigureRenderer(400, 200, scale=0.5)
    figure = renderer.render(np.zeros((200, 400, 3), np.uint8), [Box((100, 100, 200, 180), PALETTE[1])], "x")

    assert figure.shape[:2] == (100 + renderer.caption_height, 200)
    assert figure[90, 75].tolist() == _bgr(PALETTE[1])


def _write_sequence(root: Path, name: str = "MOT17-02", length: int = 4) -> None:
    seq = root / "train" / name
    (seq / "img1").mkdir(parents=True)
    (seq / "gt").mkdir()
    for f in range(1, length + 1):
        Image.new("RGB", (64, 48), (90, 90, 90)).save(seq / "img1" / f"{f:06d}.jpg")
    (seq / "gt" / "gt.txt").write_text(
        "".join(f"{f},1,{f + 5},5,20,30,1,1,{0.05 if f == 3 else 1}\n" for f in range(1, length + 1))
    )
    (seq / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={length}\n"
        "imWidth=64\nimHeight=48\nimExt=.jpg\n"
    )


def test_cli_renders_a_still(tmp_path, capsys):
    _write_sequence(tmp_path / "data")
    out = tmp_path / "fig.png"

    assert viz_cli.main(["MOT17-02", "--gt", "--frame", "3", "--root", str(tmp_path / "data"), "--out", str(out)]) == 0

    image = cv2.imread(str(out))
    assert image.shape[1] == 64 and image.shape[0] > 48
    assert "wrote" in capsys.readouterr().out


def test_cli_renders_a_video(tmp_path):
    _write_sequence(tmp_path / "data")
    out = tmp_path / "clip.mp4"

    assert viz_cli.main(["MOT17-02", "--gt", "--split", "val_half", "--root", str(tmp_path / "data"), "--out", str(out)]) == 0

    video = cv2.VideoCapture(str(out))
    count = 0
    while video.read()[0]:
        count += 1
    video.release()
    assert count == 1  # val_half of a 4-frame sequence is frame 4
