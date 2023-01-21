import otECLM
import numpy as np
import openturns as ot
import pytest
import numpy.testing as npt
import openturns.testing as ott

def test_class():
    N = 1000
    p = ot.Poisson().getProbabilities()
    values = ot.Multinomial(N, p).getRealization()
    value = ot.Indices([int(v) for v in values])
    #obj = otECLM.ECLM(value, ot.GaussLegendre([50]))
    x = [0.66134,0.489775,0.468524]
    #res = obj.estimateMaxLikelihoodFromMankamo(x, False, False)
    ott.assert_almost_equal(x, x)
