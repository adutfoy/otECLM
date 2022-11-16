"""
========================================
Example : Extended Common Load Modelling
========================================
"""
# %%
# | Import the required modules

# %%
import openturns as ot
from openturns.viewer import View
import oteclm

# %%
# ===========
# Description
# ===========
#
# We consider a common cause failure (CCF) groupe with *n=7* identical and independent components. The total impact vector of this CCF group is estimated after N=1002100 demands or tests on the group.
#
# .. math::
#
#     V_t^{n,N} = [1000000, 2000, 200, 30, 20, 5, 0, 0]
#
#

# %%
n = 7
vectImpactTotal = ot.Indices(n+1)
vectImpactTotal[0] = 1000000
vectImpactTotal[1] = 2000
vectImpactTotal[2] = 200
vectImpactTotal[3] = 30
vectImpactTotal[4] = 20
vectImpactTotal[5] = 5
vectImpactTotal[6] = 0
vectImpactTotal[7] = 0

# %%
# Create the ECLM class. We will use the Gauss Legendre quadrature algorithm to compute all the integrals of the ECLM model. The use of 50 points is sufficicient to reach a good precision.

# %%
myECLM = oteclm.ECLM(vectImpactTotal, ot.GaussLegendre([50]))

# %%
# ==============================
# Estimate the optimal parameter
# ==============================
#
# We use the Mankamo assumption. We use the maximum likelihood estimators of the *Mankamo parameter*. We want to get all the graphs of the likelihood function at the optimal Mankamo parameter.
# 
# We start by verifying that our starting point :math:`(P_x, C_{co}, c_x)` for the optimization algorithm verifies the constraints!
#

# %%
startingPoint = [5.0e-3, 0.51, 0.85]
print(myECLM.verifyMankamoConstraints(startingPoint))

# %%
# If the point is not valid, we can ask for a valid one by giving $C_x$.

startingPoint = myECLM.computeValidMankamoStartingPoint(0.7)
startingPoint

# %%
# Anyway, if the starting point is not valid, the function *estimateMaxLikelihoodFromMankamo* will automatically change it by itself.

# %%
visuLikelihood = True
mankamoParam, generalParam, finalLogLikValue, graphesCol = myECLM.estimateMaxLikelihoodFromMankamo(startingPoint, visuLikelihood, verbose=False)
print('Mankamo parameter : ', mankamoParam)
print('general parameter : ', generalParam)
print('finalLogLikValue : ', finalLogLikValue)

# %%
gl = ot.GridLayout(2,3)
for i in range(6):
    gl.setGraph(i//3, i%3, graphesCol[i])
gl

# %%
# ==============================
# Compute the ECLM probabilities
# ==============================

# %%
PEG_list = myECLM.computePEGall()
print('PEG_list = ', PEG_list)
print('')

PSG_list = myECLM.computePSGall()
print('PSG_list = ', PSG_list)
print('')

PES_list = myECLM.computePESall()
print('PES_list = ', PES_list)
print('')

PTS_list = myECLM.computePTSall()
print('PTS_list = ', PTS_list)

# %%
# ================================================
# Generate a sample of the parameters by Bootstrap
# ================================================
#
# We use the bootstrap sampling to get a sample of total impact vectors. Each total impact vector value is associated to an optimal Mankamo parameter and an optimal general parameter.
# We fix the size of the bootstrap sample.
# We also fix the number of realisations after which the sample is saved.
# Each optimisation problem is initalised with the optimal parameter found for the total impact vector.
# 
# The sample is generated and saved in a csv file.

# %%
Nbootstrap = 100
blockSize = 256

# %%
startingPoint = mankamoParam[1:4]
fileNameSampleParam = 'sampleParamFromMankamo_{}.csv'.format(Nbootstrap)
myECLM.estimateBootstrapParamSampleFromMankamo(Nbootstrap, startingPoint, blockSize, fileNameSampleParam)

# | Create the sample of all the ECLM probabilities associated to the sample of the parameters.

# %%
fileNameECLMProbabilities = 'sampleECLMProbabilitiesFromMankamo_{}.csv'.format(Nbootstrap)
myECLM.computeECLMProbabilitiesFromMankano(blockSize, fileNameSampleParam, fileNameECLMProbabilities)

# %%
# ======================================================
# Graphically analyse the bootstrap sample of parameters
# ======================================================
#
# We create the Pairs graphs of the Mankamo and general parameters.

# %%
graphPairsMankamoParam, graphPairsGeneralParam, graphMarg_list, descParam = myECLM.analyseGraphsECLMParam(fileNameSampleParam)

# %%
graphPairsMankamoParam

# %%
graphPairsGeneralParam

# | We estimate the distribution of each parameter with a Histogram and a normal kernel smoothing.

# %%
gl = ot.GridLayout(3,3)
for k in range(len(graphMarg_list)):
    gl.setGraph(k//3, k%3, graphMarg_list[k])
gl

# %%
# ==================================================================
# Graphically analyse the bootstrap sample of the ECLM probabilities
# ==================================================================
#
# We create the Pairs graphs of all the ECLM probabilities. We limit the graphical study to the multiplicities lesser than :math:`k_{max}`.

# %%
kMax = 5

graphPairs_list, graphPEG_PES_PTS_list, graphMargPEG_list, graphMargPSG_list, graphMargPES_list, graphMargPTS_list, desc_list = myECLM.analyseGraphsECLMProbabilities(fileNameECLMProbabilities, kMax)

# %%
descPairs = desc_list[0]
descPEG_PES_PTS = desc_list[1]
descMargPEG = desc_list[2]
descMargPSG = desc_list[3]
descMargPES = desc_list[4]
descMargPTS = desc_list[5]

# %%
graphPairs_list[0]

# %%
graphPairs_list[1]

# %%
graphPairs_list[2]

# %%
graphPairs_list[3]

# %%
# | Fix a k <=kMax

k = 0
graphPEG_PES_PTS_list[k]

# %%
len(graphMargPEG_list)
gl = ot.GridLayout(2,3)
for k in range(len(graphMargPEG_list)):
    gl.setGraph(k//3, k%3, graphMargPEG_list[k])
gl

# %%
gl = ot.GridLayout(2,3)
for k in range(len(graphMargPSG_list)):
    gl.setGraph(k//3, k%3, graphMargPSG_list[k])
gl    

# %%
gl = ot.GridLayout(2,3)
for k in range(len(graphMargPES_list)):
    gl.setGraph(k//3, k%3, graphMargPES_list[k])
gl    

# %%
gl = ot.GridLayout(2,3)
for k in range(len(graphMargPTS_list)):
    gl.setGraph(k//3, k%3, graphMargPTS_list[k])
gl    

# %%
# ============================================
# Fit a distribution to the ECLM probabilities
# ============================================
#
# We fit a distribution among a given list to each ECLM probability. We test it with the Lilliefors test. 
# We also compute the confidence interval of the specified level.

# %%
factoryColl = [ot.BetaFactory(), ot.LogNormalFactory(), ot.GammaFactory()]
confidenceLevel = 0.9
IC_list, graphMarg_list, descMarg_list = myECLM.analyseDistECLMProbabilities(fileNameECLMProbabilities, kMax, confidenceLevel, factoryColl)

IC_PEG_list, IC_PSG_list, IC_PES_list, IC_PTS_list = IC_list
graphMargPEG_list, graphMargPSG_list, graphMargPES_list, graphMargPTS_list = graphMarg_list
descMargPEG, descMargPSG, descMargPES, descMargPTS = descMarg_list

# %%
for k in range(len(IC_PEG_list)):
    print('IC_PEG_', k, ' = ', IC_PEG_list[k])

for k in range(len(IC_PSG_list)):
    print('IC_PSG_', k, ' = ', IC_PSG_list[k])

for k in range(len(IC_PES_list)):
    print('IC_PES_', k, ' = ', IC_PES_list[k])

for k in range(len(IC_PTS_list)):
    print('IC_PTS_', k, ' = ', IC_PTS_list[k])

# | We draw all the estimated distributions and the title gives the best model.

# %%
gl = ot.GridLayout(2,3)
for k in range(len(graphMargPEG_list)):
    gl.setGraph(k//3, k%3, graphMargPEG_list[k])
gl

# %%
gl = ot.GridLayout(2,3)
for k in range(len(graphMargPSG_list)):
    gl.setGraph(k//3, k%3, graphMargPSG_list[k])
gl

# %%
gl = ot.GridLayout(2,3)
for k in range(len(graphMargPES_list)):
        gl.setGraph(k//3, k%3, graphMargPES_list[k])
gl

# %%
gl = ot.GridLayout(2,3)
for k in range(len(graphMargPTS_list)):
    gl.setGraph(k//3, k%3, graphMargPTS_list[k])
gl

# %%
# ====================================================================================
# Analyse the minimal multiplicity which probability is greater than a given threshold
# ====================================================================================
# 
# We fix *p* and we get the minimal multiplicity :math:`k_{max}` such that :
# .. math::
#
#    k_{max} = \arg\max \{k| \mbox{PTS}(k|n) \geq p \}
#

# %%
p = 1.0e-5
nameSeuil = '10M5'

# %%
kMax = myECLM.computeKMaxPTS(p)
print('kMax = ', kMax)

# | Then we use the bootstrap sample of the Mankamo parameters to generate a sample of :math:`k_{max}`. We analyse the distribution of $k_{max}$: we estimate it with the empirical distribution and we derive a confidence interval of order :math:`90\%`.

# %%
fileNameSampleParam = 'sampleParamFromMankamo_{}.csv'.format(Nbootstrap)
fileNameSampleKmax = 'sampleKmaxFromMankamo_{}_{}.csv'.format(Nbootstrap, nameSeuil)
gKmax = myECLM.computeAnalyseKMaxSample(p, blockSize, fileNameSampleParam, fileNameSampleKmax)

# %%
gKmax
