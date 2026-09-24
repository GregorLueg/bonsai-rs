import pytest

import bonsai_rs as bs


@pytest.fixture(scope="session")
def sim() -> bs.SimulatedData:
    return bs.datasets.simulate(128, 100, seed=2)


@pytest.fixture(scope="session")
def counts() -> bs.SimulatedCounts:
    return bs.datasets.simulate_counts(64, 300, seed=1)


@pytest.fixture(scope="session")
def fitted(sim: bs.SimulatedData) -> bs.BonsaiResult:
    return bs.bonsai(sim.means, sim.sds)
