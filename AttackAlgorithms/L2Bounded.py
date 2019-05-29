import numpy as np

from AttackAlgorithms.BaseClass import AlgorithmsBaseClass
from AttackAlgorithms.AttackModels import AttackModels


class L2Bounded(AlgorithmsBaseClass):
    _L2AlgorithmIdentity = 4
    def __init__(self, fmodel, isCaffeModel, batcher, targetLabel, savePath, eta, modelChoice, beta1=0, beta2=0, attackid=None):
        AlgorithmsBaseClass._algorithmID = L2Bounded._L2AlgorithmIdentity
        AlgorithmsBaseClass._algorithmDescription = AttackModels.getAlgorithmDescription(AlgorithmsBaseClass._algorithmID)
        super().__init__(fmodel=fmodel, isCaffeModel=isCaffeModel, batcher=batcher, targetLabel=targetLabel, savePath=savePath, eta=eta, modelChoice=modelChoice,
                         beta1=beta1, beta2=beta2, attackid=attackid)

    def updatePeturbationByBackProjection(self):
        minPerturbation = np.minimum(1, self.eta / np.linalg.norm(self.perturbation))
        self.perturbation = np.multiply(self.perturbation, minPerturbation)

    def run(self, checkForAccuracy=False):
        super().run(checkForAccuracy=checkForAccuracy)