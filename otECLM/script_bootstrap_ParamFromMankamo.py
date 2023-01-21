#############################
# Ce script :
#    - lance un bootstrap sur la loi Multinomiale paramétrée par le vecteur d'impact total initial sous l'hypothèse de Mankamo
#     - calcule les estimateurs de max de vraisemblance:
#       de [Pt, Px_optim, Cco_optim, Cx_optim, pi_weight_optim, db_optim, dx_optim, dR_optim, yxm_optim] avec  yxm_optim = 1-dR_optim
#     - sauve le sample de dimension 9 dans filenameRes: de type adresse/fichier.csv
#    - le fichier fileNameInput contient les arguments: vectImpactTotal, startingPoint: de type adresse/fichier.xml


import openturns as ot
from otECLM import ECLM

from time import time
import sys

from multiprocessing import Pool
# barre de progression
import tqdm

# Ot Parallelisme desactivated
ot.TBB.Disable()

# Nombre d'échantillons bootstrap à générer
Nbootstrap = int(sys.argv[1])
# taille des blocs
blockSize = int(sys.argv[2])
# Nom du fichier csv qui contiendra le sample des paramètres (P_t, P_x, C_{co}, C_x, \pi, d_b, d_x, d_R, y_{xm})
fileNameRes = str(sys.argv[3])

print('boostrap param : ')
print('Nbootstrap, blockSize, fileNameRes  = ', Nbootstrap, blockSize, fileNameRes )


# Import de  vectImpactTotal et startingPoint
myStudy = ot.Study('myECLM.xml')
myStudy.load()
totalImpactVector = ot.Indices()
myStudy.fillObject('totalImpactVector', totalImpactVector)
startingPoint = ot.Point()
myStudy.fillObject('startingPoint', startingPoint)

myECLM = ECLM(totalImpactVector)

# Sollicitations number
N = sum(vectImpactTotal)

# Empirical distribution of the number of sets of failures among N sollicitations
MultiNomDist = ot.Multinomial(N, [v/N for v in vectImpactTotal])

def job(inP):
    point, startingPoint = inP
    vectImpactTotal = ot.Indices([int(round(x)) for x in point])
    myECLM.setTotalImpactVector(vectImpactTotal)    
    res = myECLM.estimateMaxLikelihoodFromMankamo(startingPoint, False, False)
    resMankamo = res[0]
    resGeneral = res[1]
    resFinal = resMankamo + resGeneral
    print('job : resFinal = ', resFinal)
    return resFinal

Ndone = 0
block = 0
t00 = time()


#  Si des calculs ont déjà été faits, on les importe:
try:
    print("[ParamFromMankano] Try to import previous results from {}".format(fileNameRes))
    allResults = ot.Sample.ImportFromCSVFile(fileNameRes)
    print("import ok")
except:
    print("No previous results")
    # the size of res is 9
    allResults = ot.Sample(0, 9)

allResults.setDescription(['Pt', 'Px', 'Cco', 'Cx', 'pi', 'db', 'dx', 'dR', 'yxm'])

# On passe les Nskip points déjà calculés (pas de pb si Nskip=0)
Nskip = allResults.getSize()
N_remainder = Nbootstrap - Nskip
print("Skip = ", Nskip)
for i in range(Nskip):
    noMatter = MultiNomDist.getRealization()
while Ndone < N_remainder:
    block += 1
     # Nombre de calculs qui restent à faire
    size = min(blockSize, N_remainder - Ndone)
    print("Generate bootstrap data, block=", block, "size=", size, "Ndone=", Ndone, "over", Nbootstrap)
    t0 = time()
    allInputs = [[MultiNomDist.getRealization(), startingPoint] for i in range(size)]
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



    
