from AttackAlgorithms.L2Bounded import L2Bounded as L2Algorithm
from AttackAlgorithms.LInfinityBounded import LInfinityBounded as LinfinityAlgorithm

class AttackAlgorithmFactory:
    L2_BOUNDED_ALGORITHM = L2Algorithm._L2AlgorithmIdentity
    LINF_BOUNDED_ALGORITHM = LinfinityAlgorithm._LInfinityAlgorithmIdentity

    @staticmethod
    def getAttackAlgorithm(algorithmChoice):
        algo = None
        if algorithmChoice == AttackAlgorithmFactory.L2_BOUNDED_ALGORITHM :
            algo = L2Algorithm
        elif algorithmChoice == AttackAlgorithmFactory.LINF_BOUNDED_ALGORITHM:
            algo = LinfinityAlgorithm
        return algo
