import oteclm
import openturns as ot
import openturns.testing as ott
import math as math

def test_class_2():
    n = 5
    N = 10000
    proba_PES_nonNormalisees = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    somme =0
    for p in proba_PES_nonNormalisees:
        somme += p
    PESref = [p/somme for p in proba_PES_nonNormalisees]
    vectImpactTotal = ot.Indices(n+1)
    vectImpactTotal[0] = 2897
    vectImpactTotal[1] = 2317
    vectImpactTotal[2] = 1865
    vectImpactTotal[3] = 1463
    vectImpactTotal[4] = 963
    vectImpactTotal[5] = 495
    intAlgo = ot.GaussLegendre([50])
    myECLM = oteclm.ECLM(vectImpactTotal, intAlgo)
    Pt_estim = 0.0
    for i in range(1,n+1):
        Pt_estim += i*vectImpactTotal[i]/(n*N)
    x = [0.5*Pt_estim, 0.51, 0.85]
    mankamoParam, generalParam, finalLogLikValue, graphesCol = myECLM.estimateMaxLikelihoodFromMankamo(x, False)
    
    # Comparison with exact results
    PEGref = list()
    for k in range(len(PESref)):
        PEGref.append(PESref[k]/math.comb(n, k))
    
    PEG = myECLM.computePEGall()
    PES = myECLM.computePESall()
    
    ott.assert_almost_equal(PEG, PEGref, 1e-1)
    ott.assert_almost_equal(PES, PESref, 1e-1)
