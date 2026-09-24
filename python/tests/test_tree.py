import numpy as np
import pytest

import bonsai_rs as bs


def test_newick_round_trip_keeps_distances(fitted):
    tree, labels = bs.read_newick(bs.to_newick(fitted.tree))
    # Parsed leaves come in order of appearance; the labels map them back.
    original = np.array([int(label) for label in labels])
    before = bs.tree_distances(fitted.tree)[np.ix_(original, original)]
    np.testing.assert_allclose(bs.tree_distances(tree), before, rtol=1e-9)


@pytest.mark.parametrize("kind", ["daylight", "angle", "dendrogram"])
def test_layout_covers_every_node(fitted, kind):
    xy = bs.layout(fitted.tree, kind=kind)
    assert xy.shape == (fitted.tree.n_nodes, 2)
    assert np.isfinite(xy).all()


def test_hyperbolic_layout_sits_in_the_unit_disk(fitted):
    xy = bs.layout(fitted.tree, hyperbolic=True)
    assert (np.hypot(xy[:, 0], xy[:, 1]) < 1.0).all()


def test_cluster_partitions_the_leaves(fitted):
    c = bs.cluster(fitted.tree, 4)
    assert c.sizes.sum() == fitted.tree.n_leaves
    assert len(c.centres) == 4
    assert set(np.unique(c.leaf_cluster)) == {0, 1, 2, 3}


def test_pairwise_distances_match_the_matrix(fitted):
    full = bs.tree_distances(fitted.tree)
    pairs = np.array([[0, 1], [5, 100], [127, 3]])
    np.testing.assert_allclose(
        bs.tree_distances(fitted.tree, pairs), full[pairs[:, 0], pairs[:, 1]]
    )
    assert (np.diag(full) == 0).all()


def test_out_of_range_pair_is_refused(fitted):
    with pytest.raises(ValueError, match="out of range"):
        bs.tree_distances(fitted.tree, np.array([[0, 1000]]))


def test_a_tree_is_at_distance_zero_from_itself(fitted):
    assert bs.robinson_foulds(fitted.tree, fitted.tree) == 0
