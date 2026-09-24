import json
import os

from reidtrack.data import Mot17, prepare


def run(raw, out, *extra):
    return prepare.main(["--raw", str(raw), "--out", str(out), *extra])


def test_builds_one_folder_per_sequence_with_hard_links(release, tmp_path):
    out = tmp_path / "out"

    assert run(release, out) == 0

    seq = out / "train" / "MOT17-02"
    assert os.path.samefile(seq / "img1" / "000001.jpg", release / "train" / "MOT17-02-FRCNN" / "img1" / "000001.jpg")
    assert sorted(p.name for p in (seq / "det").iterdir()) == ["DPM.txt", "FRCNN.txt", "SDP.txt"]
    assert (seq / "det" / "DPM.txt").read_text() == (release / "train" / "MOT17-02-DPM" / "det" / "det.txt").read_text()
    assert "name=MOT17-02\n" in (seq / "seqinfo.ini").read_text()
    assert not (out / "test" / "MOT17-01" / "gt").exists()


def test_copy_mode_makes_independent_files(release, tmp_path):
    out = tmp_path / "out"

    assert run(release, out, "--link-mode", "copy") == 0

    img = out / "train" / "MOT17-02" / "img1" / "000001.jpg"
    assert not os.path.samefile(img, release / "train" / "MOT17-02-FRCNN" / "img1" / "000001.jpg")
    assert img.read_bytes() == b"MOT17-02-001"


def test_exports_trackeval_ground_truth_renumbered_per_split(release, tmp_path):
    out = tmp_path / "out"
    run(release, out)
    trackeval = out / "trackeval"

    val = (trackeval / "MOT17-val_half" / "MOT17-02" / "gt" / "gt.txt").read_text().splitlines()
    assert [line.split(",")[0] for line in val] == ["1", "2"]
    assert val[0] == "1,1,5,2,10,20,1,1,1"  # original frame 5, other columns untouched
    assert "seqLength=2\n" in (trackeval / "MOT17-val_half" / "MOT17-02" / "seqinfo.ini").read_text()

    train = (trackeval / "MOT17-train_half" / "MOT17-02" / "gt" / "gt.txt").read_text().splitlines()
    assert [line.split(",")[0] for line in train] == ["1", "2", "3", "4"]

    assert (trackeval / "seqmaps" / "MOT17-val_half.txt").read_text() == "name\nMOT17-02\nMOT17-04\n"
    assert not (trackeval / "MOT17-train" / "MOT17-01").exists()


def test_manifest_records_metadata_and_split_ranges(release, tmp_path):
    out = tmp_path / "out"
    run(release, out)

    manifest = json.loads((out / "manifest.json").read_text())

    entry = manifest["sequences"]["MOT17-02"]
    assert entry["splits"] == {"train": [1, 6], "train_half": [1, 4], "val_half": [5, 6]}
    assert entry["moving_camera"] is False
    assert "splits" not in manifest["sequences"]["MOT17-01"]
    assert manifest["images_compared_by"] == "content"


def test_rerun_is_a_no_op(release, tmp_path):
    out = tmp_path / "out"

    assert run(release, out) == 0
    assert run(release, out) == 0
    assert len(list((out / "train" / "MOT17-02" / "img1").iterdir())) == 6


def test_differing_image_contents_stop_the_build(release, tmp_path, capsys):
    out = tmp_path / "out"
    (release / "train" / "MOT17-04-SDP" / "img1" / "000003.jpg").write_bytes(b"XOT17-04-003")

    assert run(release, out) == 1

    assert "000003.jpg" in capsys.readouterr().err
    assert not out.exists()


def test_size_check_misses_same_size_corruption(release, tmp_path):
    (release / "train" / "MOT17-04-SDP" / "img1" / "000003.jpg").write_bytes(b"XOT17-04-003")

    assert run(release, tmp_path / "out", "--no-hash") == 0


def test_differing_ground_truth_stops_the_build(release, tmp_path, capsys):
    (release / "train" / "MOT17-02-DPM" / "gt" / "gt.txt").write_text("1,1,1,1,1,1,1,1,1\n")

    assert run(release, tmp_path / "out") == 1
    assert "gt.txt differs" in capsys.readouterr().err


def test_missing_detector_copy_stops_the_build(release, tmp_path, capsys):
    folder = release / "train" / "MOT17-02-SDP"
    folder.rename(folder.with_name("unrelated"))

    assert run(release, tmp_path / "out") == 1
    assert "missing detector folders ['SDP']" in capsys.readouterr().err


def test_prepared_layout_loads(release, tmp_path):
    out = tmp_path / "out"
    run(release, out)

    data = Mot17(out)
    assert [s.name for s in data.sequences("train")] == ["MOT17-02", "MOT17-04"]
    seq = data.sequence("MOT17-02")
    assert seq.load_gt("val_half").frame.tolist() == [5, 6]
    assert seq.load_det("FRCNN").frame.tolist() == [1, 2]
    assert seq.image_path(3).read_bytes() == b"MOT17-02-003"
