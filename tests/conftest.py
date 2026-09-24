from pathlib import Path

import pytest

SEQINFO = """[Sequence]
name={name}
imDir=img1
frameRate=30
seqLength={length}
imWidth=64
imHeight=48
imExt=.jpg
"""


def write_release(root: Path, sequences: dict[str, int], subset: str = "train") -> Path:
    """Write a miniature MOT17 release: every sequence copied once per detector."""
    for name, length in sequences.items():
        gt = "".join(f"{f},1,{f},2,10,20,1,1,1\n" for f in range(1, length + 1))
        for detector in ("DPM", "FRCNN", "SDP"):
            folder = root / subset / f"{name}-{detector}"
            (folder / "img1").mkdir(parents=True)
            for f in range(1, length + 1):
                (folder / "img1" / f"{f:06d}.jpg").write_bytes(f"{name}-{f:03d}".encode())
            (folder / "det").mkdir()
            (folder / "det" / "det.txt").write_text(f"2,-1,1,1,5,5,0.9\n1,-1,1,1,5,5,0.{len(detector)}\n")
            if subset == "train":
                (folder / "gt").mkdir()
                (folder / "gt" / "gt.txt").write_text(gt)
            (folder / "seqinfo.ini").write_text(SEQINFO.format(name=f"{name}-{detector}", length=length))
    return root


@pytest.fixture
def release(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    write_release(raw, {"MOT17-02": 6, "MOT17-04": 5})
    write_release(raw, {"MOT17-01": 4}, subset="test")
    return raw
