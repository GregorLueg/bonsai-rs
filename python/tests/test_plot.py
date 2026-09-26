import numpy as np
import pytest

pytest.importorskip("matplotlib")


@pytest.fixture(autouse=True, scope="module")
def _agg_backend():
    import matplotlib

    matplotlib.use("Agg")


def test_plot_returns_a_figure_and_axes(fitted):
    fig, ax = fitted.plot()
    assert ax in fig.axes
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_colours_leaves_by_label(fitted):
    import matplotlib.pyplot as plt

    labels = np.arange(fitted.tree.n_leaves) % 3
    fig, ax = fitted.plot(colours=labels)
    assert len(ax.collections) >= 2  # edge LineCollection + leaf scatter
    plt.close(fig)


def test_plot_draws_into_a_given_axes(fitted):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out_fig, out_ax = fitted.plot(ax=ax)
    assert out_fig is fig
    assert out_ax is ax
    plt.close(fig)


def test_plot_rejects_mismatched_colours(fitted):
    with pytest.raises(ValueError, match="colours"):
        fitted.plot(colours=np.zeros(fitted.tree.n_leaves + 1))
