import pytest

from reidtrack.eval.latency import StageTimer


def fake_clock(*ticks):
    it = iter(ticks)
    return lambda: next(it)


def test_stages_other_and_total_per_frame():
    # frame 0 is warm-up; frame 1: a takes 1 s, b 0.5 s, 0.5 s untimed, 2 s in total
    timer = StageTimer(warmup=1, clock=fake_clock(0, 1, 1.5, 2, 3, 3, 3.5, 4))
    for stages in (["a"], ["a", "b"]):
        for name in stages:
            with timer.stage(name):
                pass
        timer.next_frame()

    s = timer.summary()

    assert timer.frames == 1
    assert s["a"]["mean"] == pytest.approx(1000)
    assert s["b"]["mean"] == pytest.approx(500)
    assert s["other"]["mean"] == pytest.approx(500)
    assert s["total"]["mean"] == pytest.approx(2000)
    assert s["a"]["share"] == pytest.approx(0.5)
    assert timer.fps() == pytest.approx(0.5)


def test_a_stage_missing_from_a_frame_counts_as_zero():
    timer = StageTimer(warmup=0, clock=fake_clock(0, 1, 1, 2, 3, 3, 4, 4))
    with timer.stage("a"):
        pass
    timer.next_frame()
    with timer.stage("a"):
        pass
    with timer.stage("b"):
        pass
    timer.next_frame()

    assert timer.summary()["b"]["mean"] == pytest.approx(500)


def test_sync_runs_around_every_stage():
    calls = []
    timer = StageTimer(warmup=0, sync=lambda: calls.append(1))
    with timer.stage("a"):
        pass
    timer.next_frame()

    assert len(calls) == 2
    assert "mean ms" in timer.table()
