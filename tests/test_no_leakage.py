"""Data leakage does not produce a visibly wrong number -- it produces a
flatteringly right one. So it is checked, not trusted."""
import pytest

from medlift3d.datasets import load_split, make_split, save_split, verify_split


def test_split_is_disjoint_and_deterministic():
    ids = [f"case{i:03d}" for i in range(40)]
    a = make_split(ids, seed=42)
    verify_split(a)
    assert make_split(ids, seed=42) == a
    assert set(a["train"]) | set(a["val"]) | set(a["test"]) == set(ids)


def test_verify_split_catches_overlap():
    with pytest.raises(AssertionError, match="LEAKAGE"):
        verify_split({"train": ["a", "b"], "val": ["b"], "test": ["c"]})
    with pytest.raises(AssertionError, match="LEAKAGE"):
        verify_split({"train": ["a"], "val": ["b"], "test": ["a"]})


def test_verify_split_rejects_empty_test():
    with pytest.raises(AssertionError, match="empty test"):
        verify_split({"train": ["a"], "val": ["b"], "test": []})


def test_split_round_trips_through_csv(tmp_path):
    ids = [f"case{i:03d}" for i in range(20)]
    s = make_split(ids, seed=7)
    save_split(tmp_path / "splits.csv", s)
    back = load_split(tmp_path / "splits.csv")
    verify_split(back)
    for k in ("train", "val", "test"):
        assert back[k] == s[k]
