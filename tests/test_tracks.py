import numpy as np

from reidtrack.data.mot import Tracks, load_gt, load_tracks, save_tracks


def test_save_writes_one_based_mot_rows_and_load_round_trips(tmp_path):
    tracks = Tracks(
        frame=np.array([2, 1], dtype=np.int32),
        track_id=np.array([7, 3], dtype=np.int32),
        xyxy=np.array([[10.5, 20.25, 30.5, 60.25], [0, 0, 4, 8]], dtype=np.float32),
        score=np.array([0.9, 1.0], dtype=np.float32),
    )
    path = tmp_path / "MOT17-02.txt"

    save_tracks(path, tracks)
    loaded = load_tracks(path)

    assert path.read_text().splitlines()[1] == "1,3,1.00,1.00,4.00,8.00,1.0000,-1,-1,-1"
    assert loaded.frame.tolist() == [1, 2]
    assert loaded.track_id.tolist() == [3, 7]
    np.testing.assert_allclose(loaded.xyxy, tracks.xyxy[::-1], atol=0.005)


def test_ground_truth_boxes_survive_the_round_trip_exactly(tmp_path):
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text("1,1,912,484,97,109,1,1,1\n2,1,915,480,97,110,1,1,0.5\n")
    gt = load_gt(gt_path)

    save_tracks(tmp_path / "out.txt", Tracks.from_gt(gt))

    assert [line.split(",")[:6] for line in (tmp_path / "out.txt").read_text().splitlines()] == [
        ["1", "1", "912.00", "484.00", "97.00", "109.00"],
        ["2", "1", "915.00", "480.00", "97.00", "110.00"],
    ]


def test_results_without_a_score_column_default_to_one(tmp_path):
    path = tmp_path / "r.txt"
    path.write_text("1,1,1,1,4,8\n")

    assert load_tracks(path).score.tolist() == [1.0]
