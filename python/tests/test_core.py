import numpy as np
import pytest
import scipy.sparse as sp

import bonsai_rs as bs


def test_recovers_the_simulated_tree(sim, fitted):
    # Measured 2026-09-24: RF 4 of a possible 250.
    assert bs.robinson_foulds(fitted.tree, sim.tree) <= 10
    assert fitted.tree.n_leaves == 128
    assert fitted.node_means.shape == (fitted.tree.n_nodes, len(fitted.features))
    assert [s.step for s in fitted.steps][-1] == "8 collapse"


def test_same_input_same_tree(sim, fitted):
    again = bs.bonsai(sim.means, sim.sds)
    assert again.loglik == fitted.loglik
    np.testing.assert_array_equal(again.tree.parent, fitted.tree.parent)


def test_storage_follows_the_input_dtype(sim):
    res = bs.bonsai(sim.means.astype(np.float32), sim.sds.astype(np.float32))
    assert res.node_means.dtype == np.float32


def test_counts_recover_the_simulated_tree(counts):
    res = bs.bonsai_from_counts(counts.counts, cell_totals=counts.cell_totals)
    assert bs.robinson_foulds(res.tree, counts.tree) == 0


def test_dense_and_sparse_counts_agree(counts):
    dense = bs.bonsai_from_counts(counts.counts, cell_totals=counts.cell_totals)
    for fmt in (sp.csr_matrix, sp.csc_matrix, sp.coo_array):
        sparse = bs.bonsai_from_counts(
            fmt(counts.counts), cell_totals=counts.cell_totals
        )
        assert sparse.loglik == dense.loglik


def test_the_chain_equals_its_steps(counts):
    chain = bs.bonsai_from_counts(counts.counts, cell_totals=counts.cell_totals)
    post = bs.sanity(counts.counts, cell_totals=counts.cell_totals)
    lik = bs.from_sanity(post.log_fold_changes, post.error_bars, post.variance)
    steps = bs.bonsai(lik.means, lik.sds, variances=lik.variances)
    assert steps.loglik == chain.loglik


def test_sanity_shapes(counts):
    post = bs.sanity(counts.counts, cell_totals=counts.cell_totals)
    assert post.log_fold_changes.shape == counts.counts.shape
    assert post.log_transcription_quotients.shape == counts.counts.shape
    assert post.variance.shape == (counts.counts.shape[1],)


def test_rejects_non_integer_counts(counts):
    with pytest.raises(ValueError, match="raw non-negative integers"):
        bs.sanity(np.log1p(counts.counts))


def test_rejects_non_positive_sds(sim):
    sds = sim.sds.copy()
    sds[0, 0] = 0.0
    with pytest.raises(ValueError, match="Non-positive standard deviation"):
        bs.bonsai(sim.means, sds)


def test_rejects_mismatched_shapes(sim):
    with pytest.raises(ValueError, match="sds is"):
        bs.bonsai(sim.means, sim.sds[:, :-1])


def test_rejects_an_unknown_start(sim):
    from beartype.roar import BeartypeCallHintParamViolation

    with pytest.raises(BeartypeCallHintParamViolation):
        bs.bonsai(sim.means, sim.sds, start="random")  # ty: ignore[invalid-argument-type]


def test_an_impossible_threshold_is_a_bonsai_error(sim):
    with pytest.raises(bs.BonsaiError):
        bs.bonsai(sim.means, sim.sds, min_signal_to_noise=1e12)


def test_backbone(sim):
    res = bs.backbone(sim.means, sim.sds, backbone_cells=32)
    assert bs.robinson_foulds(res.tree, sim.tree) <= 10
