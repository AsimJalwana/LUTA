import numpy as np

from AttackAlgorithms.BaseClass import AlgorithmsBaseClass
from AttackAlgorithms.AttackModels import AttackModels


class LInfinityBounded(AlgorithmsBaseClass):
    _LInfinityAlgorithmIdentity = 3
    def __init__(self, fmodel, isCaffeModel, batcher, targetLabel, savePath, eta, modelChoice, beta1=0, beta2=0, attackid=None):
        AlgorithmsBaseClass._algorithmID = LInfinityBounded._LInfinityAlgorithmIdentity
        AlgorithmsBaseClass._algorithmDescription = AttackModels.getAlgorithmDescription(AlgorithmsBaseClass._algorithmID)
        super().__init__(fmodel=fmodel, isCaffeModel=isCaffeModel, batcher=batcher, targetLabel=targetLabel, savePath=savePath, eta=eta, modelChoice=modelChoice,
                         beta1=beta1, beta2=beta2, attackid=attackid)

    def updatePeturbationByBackProjection(self):
        signPerturbation = np.sign(self.perturbation)
        minPerturbation = np.minimum(np.abs(self.perturbation), self.eta)
        self.perturbation = np.multiply(signPerturbation, minPerturbation)

    def run(self, checkForAccuracy=False):
        super().run(checkForAccuracy=checkForAccuracy)
