import pytest

from reidtrack.data.splits import frame_range


@pytest.mark.parametrize(("length", "train_last"), [(600, 301), (525, 263), (837, 419), (6, 4)])
def test_half_split_matches_centertrack(length, train_last):
    train = frame_range("train_half", length)
    val = frame_range("val_half", length)

    assert (train.first, train.last) == (1, train_last)
    assert (val.first, val.last) == (train_last + 1, length)
    assert len(train) + len(val) == length


def test_offset_renumbers_from_one():
    val = frame_range("val_half", 600)

    assert val.first - val.offset == 1
    assert 302 in val and 301 not in val


def test_full_train_split_covers_the_sequence():
    assert (frame_range("train", 600).first, frame_range("train", 600).last) == (1, 600)


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        frame_range("test_half", 10)
