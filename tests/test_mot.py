import numpy as np
import pytest

from reidtrack.data.mot import MotClass, load_det, load_gt, read_seqinfo


def test_load_gt_sorts_and_converts_to_zero_based_xyxy(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("2,1,11,21,10,20,1,1,0.5\n1,2,1,1,4,8,0,7,1\n1,1,1,1,4,8,1,1,1\n")

    gt = load_gt(path)

    assert gt.frame.tolist() == [1, 1, 2]
    assert gt.track_id.tolist() == [1, 2, 1]
    np.testing.assert_allclose(gt.xyxy[0], [0, 0, 4, 8])
    np.testing.assert_allclose(gt.xyxy[2], [10, 20, 20, 40])
    assert gt.considered.tolist() == [True, False, True]
    assert gt.cls[1] == MotClass.STATIC_PERSON


def test_targets_keeps_considered_pedestrians_only(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("1,1,1,1,4,8,1,1,1\n1,2,1,1,4,8,0,7,1\n1,3,1,1,4,8,0,1,1\n2,1,1,1,4,8,1,1,0.2\n")

    targets = load_gt(path).targets()

    assert targets.track_id.tolist() == [1, 1]
    assert targets.frame.tolist() == [1, 2]


def test_load_det_sorts_by_frame_and_accepts_dpm_columns(tmp_path):
    path = tmp_path / "det.txt"
    path.write_text("3,-1,1,1,2,2,4.5,-1,-1,-1\n1,-1,5,5,2,2,-0.5,-1,-1,-1\n")

    det = load_det(path)

    assert det.frame.tolist() == [1, 3]
    np.testing.assert_allclose(det.score, [-0.5, 4.5])
    np.testing.assert_allclose(det.xyxy[0], [4, 4, 6, 6])


def test_empty_file_gives_empty_tables(tmp_path):
    path = tmp_path / "det.txt"
    path.write_text("")

    det = load_det(path)

    assert len(det) == 0
    assert det.xyxy.shape == (0, 4)


def test_too_few_columns_is_an_error(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("1,1,1,1,4,8\n")

    with pytest.raises(ValueError, match="at least 9 columns"):
        load_gt(path)


def test_in_frames_is_inclusive(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text("".join(f"{f},1,1,1,4,8,1,1,1\n" for f in range(1, 7)))

    assert load_gt(path).in_frames(2, 4).frame.tolist() == [2, 3, 4]


def test_seqinfo_round_trip(tmp_path):
    path = tmp_path / "seqinfo.ini"
    path.write_text(
        "[Sequence]\nname=MOT17-05-DPM\nimDir=img1\nframeRate=14\nseqLength=837\n"
        "imWidth=640\nimHeight=480\nimExt=.jpg\n"
    )

    info = read_seqinfo(path)
    assert (info.name, info.frame_rate, info.length, info.width, info.height) == ("MOT17-05-DPM", 14, 837, 640, 480)

    path.write_text(info.renamed("MOT17-05").to_ini())
    assert read_seqinfo(path) == info.renamed("MOT17-05")
