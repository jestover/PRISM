"""Tests for the token trie branch point detection."""

from prism.core.token_trie import LabelTokenTrie


def test_docstring_example():
    """The docstring example should produce exactly the documented branch points."""
    labels = {
        "label1": [1, 3, 7],
        "label2": [2, 4, 3, 6],
        "label3": [2, 4, 3, 8],
        "label4": [2, 5, 1, 7],
    }

    trie = LabelTokenTrie(labels)

    assert len(trie.branch_points) == 3

    # Branch point 1: prefix=[], first token diverges
    bp1 = trie.branch_points[0]
    assert bp1.prefix == []
    assert bp1.branches == {1: ["label1"], 2: ["label2", "label3", "label4"]}

    # Branch point 2: prefix=[2], second token diverges within the 2-branch
    bp2 = trie.branch_points[1]
    assert bp2.prefix == [2]
    assert bp2.branches == {4: ["label2", "label3"], 5: ["label4"]}

    # Branch point 3: prefix=[2, 4, 3], fourth token diverges
    bp3 = trie.branch_points[2]
    assert bp3.prefix == [2, 4, 3]
    assert bp3.branches == {6: ["label2"], 8: ["label3"]}


def test_no_branching():
    """Labels with identical prefixes that only diverge at the end."""
    labels = {
        "a": [1, 2, 3],
        "b": [1, 2, 4],
    }
    trie = LabelTokenTrie(labels)

    assert len(trie.branch_points) == 1
    assert trie.branch_points[0].prefix == [1, 2]
    assert trie.branch_points[0].branches == {3: ["a"], 4: ["b"]}


def test_all_diverge_immediately():
    """Every label has a unique first token — single branch point at root."""
    labels = {
        "x": [10],
        "y": [20],
        "z": [30],
    }
    trie = LabelTokenTrie(labels)

    assert len(trie.branch_points) == 1
    bp = trie.branch_points[0]
    assert bp.prefix == []
    assert bp.branches == {10: ["x"], 20: ["y"], 30: ["z"]}


def test_single_label():
    """A single label produces no branch points."""
    labels = {"only": [1, 2, 3]}
    trie = LabelTokenTrie(labels)
    assert len(trie.branch_points) == 0
