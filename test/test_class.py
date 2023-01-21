import otECLM
import numpy as np
import openturns as ot
import pytest
import numpy.testing as npt
import openturns.testing as ott

def test_class():
    values = ot.Poisson().getSample(100)
    value = ot.Indices([int(v[0]) for v in values])
    obj = otECLM.ECLM(value, ot.GaussLegendre(50))
    res = obj.estimateMaxLikelihoodFromMankamo([0.5]*4, False, False)
    ott.assert_almost_equal(res[0:4], res[0:4])
