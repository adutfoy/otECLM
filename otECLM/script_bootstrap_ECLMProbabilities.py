#############################
# Ce script :
#    - récupère un échantillon de (Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim] dans le fichier fileNameInput (de type adresse/fichier.csv)
#     - calcule toutes les grandeurs probabilistes de l'ECLM
#     - sauve le sample des grandeurs  probabilistes dans  le fichier fileNameRes (de type adresse/fichier.csv)



import openturns as ot
from otECLM import ECLM

from time import time
import sys

from multiprocessing import Pool
# barre de progression
import tqdm

# Ot Parallelisme desactivated
ot.TBB.Disable()


# Nombre de grappes
n = int(sys.argv[1])
# taille des blocs
blockSize = int(sys.argv[2])
# Nom du fichier csv contenant le sample des paramètres (P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})
fileNameInput = str(sys.argv[3])
# Nom du fichier de sauvegarde: de type adresse/fichier.csv
fileNameRes = str(sys.argv[4])

print('ECLM prob')
print('n, fileNameInput, fileNameRes = ', n, fileNameInput, fileNameRes)

# Import de  vectImpactTotal et startingPoint
myStudy = ot.Study('myECLM.xml')
myStudy.load()
totalImpactVector = ot.Indices()
myStudy.fillObject('totalImpactVector', totalImpactVector)

myECLM = ECLM(totalImpactVector)


def job(inP):
    # inP = [Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim]
    myECLM.setGeneralParameter(inP[4:9])
    PEG_list = myECLM.computePEGall()
    PSG_list = myECLM.computePSGall()
    PES_list = myECLM.computePESall()
    PTS_list = myECLM.computePTSall()
    res_list = list()
    res_list += PEG_list
    res_list += PSG_list
    res_list += PES_list
    res_list += PTS_list
    return res_list


Ndone = 0
block = 0
t00 = time()


# Import de l'échantillon de [Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim]
# sauvé dans le fichier fileNameInput
sampleParam = ot.Sample.ImportFromCSVFile(fileNameInput)
Nsample = sampleParam.getSize()


#  Si des calculs ont déjà été faits, on les importe:
try:
    print("[ECLMProbabilities] Try to import previous results from {}".format(fileNameRes))
    allResults = ot.Sample.ImportFromCSVFile(fileNameRes)
    print("import ok")
except:
    # la dimension du sample est 4*(n+1)
    dim = 4*(n+1)
    allResults = ot.Sample(0, dim)

    
# Description
desc = ot.Description()
desc =  ['PEG('+str(k) + '|' +str(n) +')' for k in range(n+1)]
desc += ['PSG('+str(k) +')' for k in range(n+1)]
desc += ['PES('+str(k) + '|' +str(n) +')' for k in range(n+1)]
desc += ['PTS('+str(k) + '|' +str(n) +')' for k in range(n+1)]
allResults.setDescription(desc) 

# On passe les Nskip points déjà calculés (pas de pb si Nskip=0)
Nskip = allResults.getSize()
remainder_sample = sampleParam.split(Nskip)
N_remainder = remainder_sample.getSize()
print('N_remainder = ', N_remainder)
while Ndone < N_remainder:
    block += 1
    # Nombre de calculs qui restent à faire
    size = min(blockSize, N_remainder - Ndone)
    print("Generate bootstrap data, block=", block, "size=", size, "Ndone=", Ndone, "over", N_remainder)
    t0 = time()
    allInputs = [remainder_sample[(block-1)*blockSize + i] for i in range(size)]
    t1 = time()
    print("t=%.3g" % (t1 - t0), "s")

    t0 = time()
    pool = Pool()
    # Calcul parallèle: pas d'ordre, retourné dès que réalisé
    allResults.add(list(tqdm.tqdm(pool.imap_unordered(job, allInputs), total=len(allInputs))))
    pool.close()
    t1 = time()
    print("t=%.3g" % (t1 - t0), "s", "t (start)=%.3g" %(t1 - t00), "s")
    Ndone += size
    # Sauvegarde apres chaque bloc
    allResults.exportToCSVFile(fileNameRes)
