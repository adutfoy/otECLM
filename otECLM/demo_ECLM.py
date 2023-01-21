from openturns import *
from otECLM import *
from openturns.viewer import View
from time import time

Log.Show(Log.NONE)
###########################
# Données

# Nombre de grappes
n=53

# vecteur d'impact total (valeurs entière)
vectImpactTotal = Indices(n+1)
vectImpactTotal[0] = 2794644
vectImpactTotal[1] = 2032
vectImpactTotal[2] = 172
vectImpactTotal[3] = 33
vectImpactTotal[4] = 22
vectImpactTotal[5] = 53
vectImpactTotal[6] = 22
vectImpactTotal[7] = 11
vectImpactTotal[9] = 11


myECLM = ECLM(vectImpactTotal)

print('Size of the CCF group = ', myECLM.getN())

###################################################
# Estimate (Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim)

print('Estimation param')
print('================')
print('Integration algorithm: ', myECLM.getIntegrationAlgorithm())
print('')

visuLikelihood = False
startingPoint = [5.7e-7, 0.51, 0.85]
mankamoParam, generalParam, finalLogLikValue, graphesCol = myECLM.estimateMaxLikelihoodFromMankamo(startingPoint, visuLikelihood, verbose=False)

print('LogLikelihood = ', finalLogLikValue)
print('mankamoParam = ', mankamoParam)
print('generalParam = ', generalParam)
print('')
if visuLikelihood:
    view1 = View(graphesCol[0])
    view2 = View(graphesCol[1])
    view3 = View(graphesCol[2])
    view4 = View(graphesCol[3])
    view5 = View(graphesCol[4])
    view6 = View(graphesCol[5])
    view1.save('Figures/graphe_MaxLik_fixedlogPxCco.pdf')
    view1.save('Figures/graphe_MaxLik_fixedlogPxCco.png')
    view2.save('Figures/graphe_MaxLik_fixedlogPxCx.pdf')
    view2.save('Figures/graphe_MaxLik_fixedlogPxCx.png')
    view3.save('Figures/graphe_MaxLik_fixedCcoCx.pdf')
    view3.save('Figures/graphe_MaxLik_fixedCcoCx.png')
    view4.save('Figures/graphe_MaxLik_fixedCx.pdf')
    view4.save('Figures/graphe_MaxLik_fixedCx.png')
    view5.save('Figures/graphe_MaxLik_fixedCco.pdf')
    view5.save('Figures/graphe_MaxLik_fixedCco.png')
    view6.save('Figures/graphe_MaxLik_fixedlogPx.pdf')
    view6.save('Figures/graphe_MaxLik_fixedlogPx.png')
    
# Calcul des probabilités ECLM
print('calcul des probabilités ECLM')
print('============================')


#PEG_list = myECLM.computePEGall()
#print('PEG_list = ', PEG_list)

#PSG_list = myECLM.computePSGall()
#print('PSG_list = ', PSG_list)

#PES_list = myECLM.computePESall()
#print('PES_list = ', PES_list)

#PTS_list = myECLM.computePTSall()
#print('PTS_list = ', PTS_list)

print('')


###################################################
# Bootstrap sur les paramètres de Mankamo
print('Bootstrap sur les données: calcul des paramètres')
print('================================================')
Nbootstrap = 5
blockSize = 256

# startingPoint = (Px_optim, Cco_optim, Cx_optim)==> mankamoParam[1:4]
startingPoint = mankamoParam[1:4]
print('StartingPoint (Px_optim, Cco_optim, Cx_optim) = ', startingPoint)
fileNameSampleParam = 'Figures/sampleParamFromMankamo_{}.csv'.format(Nbootstrap)
myECLM.estimateBootstrapParamSampleFromMankamo(Nbootstrap, startingPoint, blockSize, fileNameSampleParam)

# Echantillon des grandeurs ECLM
print('Bootstrap sur les données: calcul des probabilités ECLM')
fileNameECLMProbabilities = 'Figures/sampleECLMProbabilitiesFromMankamo_{}.csv'.format(Nbootstrap)
myECLM.computeECLMProbabilitiesFromMankano(blockSize, fileNameSampleParam, fileNameECLMProbabilities)
print('')


#######################################################
# Analyse graphique des echantillons des Paramètres
print('Analyse graphique des echantillons des Paramètres')
print('=================================================')

graphPairsMankamoParam, graphPairsGeneralParam, graphMarg_list, descParam = myECLM.analyse_graphsECLMParam(fileNameSampleParam)

view = View(graphPairsMankamoParam)
view.save('Figures/graphe_{}_Mankamo_{}.pdf'.format("PairsMankamoParam", Nbootstrap))
view.save('Figures/graphe_{}_Mankamo_{}.png'.format("PairsMankamoParam", Nbootstrap))

view = View(graphPairsGeneralParam)
view.save('Figures/graphe_{}_Mankamo_{}.pdf'.format("GeneralParam", Nbootstrap))
view.save('Figures/graphe_{}_Mankamo_{}.png'.format("GeneralParam", Nbootstrap))

for k in range(len(graphMarg_list)):
    g = graphMarg_list[k]
    view = View(g)
    # view.show()
    fileName = 'Figures/graphe_{}_Mankamo_{}.pdf'.format(descParam[k], Nbootstrap)
    fileName = 'Figures/graphe_{}_Mankamo_{}.png'.format(descParam[k], Nbootstrap)
    view.save(fileName)

print('')

#######################################################
# Analyse graphique des echantillons des Probabilités
print('Analyse graphique des echantillons des Probabilités')
print('===================================================')

kMax = 5

graphPairs_list, graphPEG_PES_PTS_list, graphMargPEG_list, graphMargPSG_list, graphMargPES_list, graphMargPTS_list, desc_list = myECLM.analyse_graphsECLMProbabilities(fileNameECLMProbabilities, kMax)


descPairs = desc_list[0]
descPEG_PES_PTS = desc_list[1]
descMargPEG = desc_list[2]
descMargPSG = desc_list[3]
descMargPES = desc_list[4]
descMargPTS = desc_list[5]

for k in range(len(graphPairs_list)):
    view = View(graphPairs_list[k])
    # view.show()
    fileName = 'Figures/{}_{}.pdf'.format(descPairs[k], Nbootstrap)
    fileName = 'Figures/{}_{}.png'.format(descPairs[k], Nbootstrap)
    print(fileName)
    view.save(fileName)
    
for k in range(len(graphPEG_PES_PTS_list)):
    view = View(graphPEG_PES_PTS_list[k])
    # view.show()
    fileName = 'Figures/{}_{}.pdf'.format(descPEG_PES_PTS[k], Nbootstrap)
    fileName = 'Figures/{}_{}.png'.format(descPEG_PES_PTS[k], Nbootstrap)
    print(fileName)
    view.save(fileName)


for k in range(len(graphMargPEG_list)):
    view = View(graphMargPEG_list[k])
    # view.show()
    fileName = 'Figures/{}_{}.pdf'.format(descMargPEG[k], Nbootstrap)
    fileName = 'Figures/{}_{}.png'.format(descMargPEG[k], Nbootstrap)
    print(fileName)
    view.save(fileName)
    
for k in range(len(graphMargPSG_list)):
    view = View(graphMargPSG_list[k])
    # view.show()
    fileName = 'Figures/{}_{}.pdf'.format(descMargPSG[k], Nbootstrap)
    fileName = 'Figures/{}_{}.png'.format(descMargPSG[k], Nbootstrap)
    print(fileName)
    view.save(fileName)
    
for k in range(len(graphMargPES_list)):
    view = View(graphMargPES_list[k])
    # view.show()
    fileName = 'Figures/{}_{}.pdf'.format(descMargPES[k], Nbootstrap)
    fileName = 'Figures/{}_{}.png'.format(descMargPES[k], Nbootstrap)
    print(fileName)
    view.save(fileName)
    
for k in range(len(graphMargPTS_list)):
    view = View(graphMargPTS_list[k])
    # view.show()
    fileName = 'Figures/{}_{}.pdf'.format(descMargPTS[k], Nbootstrap)
    fileName = 'Figures/{}_{}.png'.format(descMargPTS[k], Nbootstrap)
    print(fileName)
    view.save(fileName)

print('')
    
#######################################################
# Analyse quantitative des echantillons des Probabilités


print('Analyse quantitative des echantillons des Probabilités')
print('======================================================')
confidenceLevel = 0.9
factoryColl = [BetaFactory(), LogNormalFactory(), GammaFactory()]

fileNameECLMProbabilities = 'Figures/sampleECLMProbabilitiesFromMankamo_200.csv'

IC_list, graphMarg_list, descMarg_list = myECLM.analyse_distECLMProbabilities(fileNameECLMProbabilities, kMax, confidenceLevel, factoryColl)

IC_PEG_list, IC_PSG_list, IC_PES_list, IC_PTS_list = IC_list
graphMargPEG_list, graphMargPSG_list, graphMargPES_list, graphMargPTS_list = graphMarg_list
descMargPEG, descMargPSG, descMargPES, descMargPTS = descMarg_list

for k in range(len(IC_PEG_list)):
    print('IC_PEG_', k, ' = ', IC_PEG_list[k])

for k in range(len(IC_PSG_list)):
    print('IC_PSG_', k, ' = ', IC_PSG_list[k])

for k in range(len(IC_PES_list)):
    print('IC_PES_', k, ' = ', IC_PES_list[k])

for k in range(len(IC_PTS_list)):
    print('IC_PTS_', k, ' = ', IC_PTS_list[k])

for k in range(len(graphMargPEG_list)):
    view = View(graphMargPEG_list[k])
    # view.show()
    fileName = 'Figures/{}_fittingTest_{}.pdf'.format(descMargPEG[k], Nbootstrap)
    fileName = 'Figures/{}_fittingTest_{}.png'.format(descMargPEG[k], Nbootstrap)
    print(fileName)
    view.save(fileName)

for k in range(len(graphMargPSG_list)):
    view = View(graphMargPSG_list[k])
    # view.show()
    fileName = 'Figures/{}_fittingTest_{}.pdf'.format(descMargPSG[k], Nbootstrap)
    fileName = 'Figures/{}_fittingTest_{}.png'.format(descMargPSG[k], Nbootstrap)
    print(fileName)
    view.save(fileName)

for k in range(len(graphMargPES_list)):
    view = View(graphMargPES_list[k])
    # view.show()
    fileName = 'Figures/{}_fittingTest_{}.pdf'.format(descMargPES[k], Nbootstrap)
    fileName = 'Figures/{}_fittingTest_{}.png'.format(descMargPES[k], Nbootstrap)
    print(fileName)
    view.save(fileName)

for k in range(len(graphMargPTS_list)):
    view = View(graphMargPTS_list[k])
    # view.show()
    fileName = 'Figures/{}_fittingTest_{}.pdf'.format(descMargPTS[k], Nbootstrap)
    fileName = 'Figures/{}_fittingTest_{}.png'.format(descMargPTS[k], Nbootstrap)
    print(fileName)
    view.save(fileName)
 
#######################################################
# Echantillon de kMax tel que
# kMax = argmax {k | PTS(k|n) > p

print('Analyse du kMax')
print('===============')

p = 1.0e-5
nameSeuil = '10M5'

kMax = myECLM.computeKMax_PTS(p)
print('KMax(', p, ') = ', kMax)
print('')



#fileNameSampleKmax = 'Resultats/sampleKmaxFromMankamo_{}_{}.csv'.format(Nbootstrap, nameSeuil)
fileNameSampleParam = 'Figures/sampleParamFromMankamo_200.csv'
fileNameSampleKmax = 'Figures/sampleKmaxFromMankamo_{}_{}.csv'.format(200, nameSeuil)

gKmax = myECLM.computeAnalyseKMaxSample(p, blockSize, fileNameSampleParam, fileNameSampleKmax)

view = View(gKmax)
view.show()
view.save('Figures/Kmax_{}_{}.png'.format(Nbootstrap, nameSeuil))
view.save('Figures/Kmax_{}_{}.pdf'.format(Nbootstrap, nameSeuil))

