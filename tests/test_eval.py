from dataclasses import replace

import numpy as np
import pytest

from reidtrack.data import Mot17, Tracks
from reidtrack.eval import __main__ as eval_cli
from reidtrack.eval.metrics import evaluate


def oracle(root, names):
    data = Mot17(root)
    return {n: Tracks.from_gt(data.sequence(n).load_gt().targets()) for n in names}


def test_ground_truth_scores_100(prepared):
    ev = evaluate(oracle(prepared, ["MOT17-02", "MOT17-04"]), "val_half", prepared)

    assert set(ev.sequences) == {"MOT17-02", "MOT17-04"}
    for scores in (*ev.sequences.values(), ev.combined):
        assert scores.hota == pytest.approx(100)
        assert scores.mota == pytest.approx(100)
        assert scores.idf1 == pytest.approx(100)
        assert scores.idsw == 0


def test_an_id_switch_is_penalised(prepared):
    results = oracle(prepared, ["MOT17-02", "MOT17-04"])
    t = results["MOT17-02"]
    results["MOT17-02"] = replace(t, track_id=np.where(t.frame == 6, 99, t.track_id).astype(np.int32))

    ev = evaluate(results, "val_half", prepared)

    assert ev.sequences["MOT17-02"].idsw == 1
    assert ev.sequences["MOT17-02"].idf1 == pytest.approx(50)
    assert ev.sequences["MOT17-04"].idf1 == pytest.approx(100)
    assert ev.combined.assa < 100


def test_rows_outside_the_split_are_ignored(prepared):
    # train_half frames of MOT17-02 are 1..4; a stray box at frame 2 must not count against val_half
    results = oracle(prepared, ["MOT17-02", "MOT17-04"])
    t = results["MOT17-02"]
    stray = Tracks(np.array([2], np.int32), np.array([5], np.int32), np.array([[0, 0, 9, 9]], np.float32), np.ones(1, np.float32))
    results["MOT17-02"] = Tracks(*(np.concatenate([a, b]) for a, b in zip(
        (t.frame, t.track_id, t.xyxy, t.score), (stray.frame, stray.track_id, stray.xyxy, stray.score))))

    assert evaluate(results, "val_half", prepared).combined.hota == pytest.approx(100)


def test_missing_sequences_are_reported(prepared):
    with pytest.raises(KeyError, match="MOT17-04"):
        evaluate(oracle(prepared, ["MOT17-02"]), "val_half", prepared)


def test_cli_prints_a_table_and_writes_json(prepared, tmp_path, capsys):
    out = tmp_path / "scores.json"

    assert eval_cli.main(["--oracle", "--split", "val_half", "--root", str(prepared), "--json", str(out)]) == 0

    text = capsys.readouterr().out
    assert text.splitlines()[0] == "MOT17 val_half, ground truth"
    assert "combined" in text and "100.0" in text
    assert '"combined"' in out.read_text()


def test_cli_reads_result_files(prepared, tmp_path, capsys):
    from reidtrack.data.mot import save_tracks

    for name, tracks in oracle(prepared, ["MOT17-02", "MOT17-04"]).items():
        save_tracks(tmp_path / "run" / f"{name}.txt", tracks)

    assert eval_cli.main(["--results", str(tmp_path / "run"), "--root", str(prepared)]) == 0
    assert "MOT17 val_half, run" in capsys.readouterr().out

    (tmp_path / "run" / "MOT17-04.txt").unlink()
    assert eval_cli.main(["--results", str(tmp_path / "run"), "--root", str(prepared)]) == 1
