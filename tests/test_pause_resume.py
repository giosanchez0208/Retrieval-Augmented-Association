import json

import numpy as np
import pytest

from reidtrack.pausing import PAUSED, PauseRequest
from reidtrack.retrieval import train as trainer
from reidtrack.retrieval.crops import crop_paths


def write_crops(root, split, id_offset, people=4, crops=4):
    rows = [(s, p + id_offset, f) for s in range(2) for p in range(1, people + 1) for f in range(1, crops + 1)]
    seq, tid, frame = (np.array(c) for c in zip(*rows))
    images_path, meta_path = crop_paths(root, split)
    images_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(id_offset)
    np.save(images_path, rng.integers(0, 255, size=(len(rows), 3, 64, 32), dtype=np.uint8))
    np.savez(meta_path, names=np.array(["MOT17-02", "MOT17-04"]), sequence=seq.astype(np.int16),
             track_id=tid.astype(np.int32), frame=(frame * 60).astype(np.int32), seconds=(frame * 2.0).astype(np.float32),
             visibility=np.ones(len(rows), np.float32))


@pytest.fixture
def crops_root(tmp_path):
    write_crops(tmp_path, "train_half", 0)  # 32 crops: 4 steps per epoch of 4 people x 2 crops
    write_crops(tmp_path, "val_half", 100)
    return tmp_path


def args(root, out, *extra):
    return ["--backbone", "osnet_x0_25", "--device", "cpu", "--people", "4", "--crops", "2", "--max-steps", "6",
            "--eval-every", "1", "--warmup", "0.5", "--root", str(root), "--out", str(out), "--name", "run", *extra]


class PauseAtThirdStep:
    def __init__(self, run_dir):
        self.checks = 0

    def __bool__(self):
        self.checks += 1
        return self.checks >= 3

    def close(self):
        pass


def test_a_paused_run_resumes_where_it_stopped(crops_root, tmp_path, monkeypatch, capsys):
    out = tmp_path / "weights"
    monkeypatch.setattr(trainer, "PauseRequest", PauseAtThirdStep)

    assert trainer.main(args(crops_root, out)) == PAUSED
    run = out / "run"
    assert (run / "resume.pt").is_file() and not (run / "last.pt").exists()
    assert "paused at epoch 1, step 3 of 6" in capsys.readouterr().out

    monkeypatch.setattr(trainer, "PauseRequest", PauseRequest)
    assert trainer.main(args(crops_root, out, "--resume", "--lr", "1")) == 0  # --lr is ignored: the schedule is kept

    assert (run / "last.pt").is_file() and not (run / "resume.pt").exists()
    history = json.loads((run / "history.json").read_text())
    assert [r["steps"] for r in history["history"]] == [4, 6]
    assert history["args"]["lr"] == pytest.approx(3.5e-4)
    assert "resuming run at epoch 1, step 3 of 6" in capsys.readouterr().out


def test_resume_without_a_saved_state_is_an_error(crops_root, tmp_path, capsys):
    assert trainer.main(args(crops_root, tmp_path / "weights", "--resume")) == 1
    assert "nothing to resume" in capsys.readouterr().err


def test_a_pause_file_requests_a_pause(tmp_path):
    request = PauseRequest(tmp_path)
    assert not request
    (tmp_path / "PAUSE").touch()
    assert request
    request.close()
    assert not (tmp_path / "PAUSE").exists()


def test_training_can_be_restricted_to_some_sequences(crops_root, tmp_path):
    out = tmp_path / "weights"

    assert trainer.main(args(crops_root, out, "--sequences", "MOT17-04", "--max-steps", "2")) == 0
    assert trainer.main(args(crops_root, out, "--sequences", "MOT17-99")) == 1

    import torch

    state = torch.load(out / "run" / "last.pt", weights_only=True)
    assert state["num_classes"] == 4  # only the four people of MOT17-04
