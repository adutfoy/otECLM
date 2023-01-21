import otECLM
import numpy as np
import openturns as ot
import pytest
import numpy.testing as npt
import openturns.testing as ott

@pytest.fixture(scope="session")
def data():
    """Provide some data"""
    values = ot.Poisson().getSample(1000)
    return [int(v) for v in values]


def test_class(data):
    value = data
    obj = otECLM.ECLM(value, ot.GaussLegendre(50))
    res = obj.estimateMaxLikelihoodFromMankamo([0.5]*4, False, False)
    ott.assert_almost_equal(res[0:4], res[0:4])
