import os

from foolbox.models import KerasModel

from AttackAlgorithms.AttackAlgorithmFactory import AttackAlgorithmFactory
from AttackAlgorithms.AttackModels import AttackModels
from Batch.BatchManager import Batcher
from ConfigManagement.ConfigManager import Manager

pathManager = Manager()
SAVE_PATH = pathManager.getSavePath()
ETA_CHOICES = [int(eta.strip()) for eta in pathManager.getEtas().split(",")]
WNIDList = [wnid.strip() for wnid in pathManager.getSourceIdentities().split(",")]
TargetWnid = [wnid.strip() for wnid in pathManager.getTargetIdentities().split(",")]
MODEL_CHOICES = [int(choice) for choice in pathManager.getAttackModels().split(",")]
ALGORITHM_CHOICES = [int(choice) for choice in pathManager.getAlgorithmIDs().split(",")]

attackID = 1

for modelChoice in MODEL_CHOICES:
    model, targetSize, isCaffeModel, decode_predictions, preprocess_input = AttackModels.getModel(modelChoice)
    Decoder = AttackModels.getModelDecoder(modelChoice)

    TargetList = []
    for wnid in TargetWnid:
        TargetList.append(Decoder.decodeWnIdToDeepModelId(wnid))

    for idx, algorithmChoice in enumerate(ALGORITHM_CHOICES):
        algorithm = AttackAlgorithmFactory.getAttackAlgorithm(algorithmChoice)

        for i in range(0, len(WNIDList)):
            WNID = WNIDList[i]
            TARGET_CLASS = int(TargetList[i])
            PATH_TO_TRAINING_IMAGES = os.path.join(pathManager.getDatasetPath(), WNID,"training/")
            PATH_TO_TESTING_IMAGES = os.path.join(pathManager.getDatasetPath(), WNID,"testing/")

            fmodel = KerasModel(model, bounds=(0, 255), predicts='logits')
            batcher = Batcher(targetSize=targetSize, preProcessInput=preprocess_input, pathOfImages=PATH_TO_TRAINING_IMAGES,
                              WnID=WNID, pathOfTestingImages=PATH_TO_TESTING_IMAGES, modelChoice=modelChoice, pathManager=pathManager)

            # Run 100 iterations without inhibition
            attackAlgo = algorithm(fmodel=fmodel, isCaffeModel=isCaffeModel, batcher=batcher,
                                   targetLabel=TARGET_CLASS, savePath=SAVE_PATH, eta=ETA_CHOICES[idx],
                                   modelChoice=modelChoice,
                                   beta1=0.9, beta2=0.999)
            attackAlgo.run()
            attackAlgo.closeLogger()

            # Max 1000 iterations with inhibition
            for j in range(0, 10):
                attackAlgo = algorithm(fmodel=fmodel, isCaffeModel=isCaffeModel, batcher=batcher,
                                              targetLabel=TARGET_CLASS, savePath=SAVE_PATH, eta=ETA_CHOICES[idx],
                                              modelChoice=modelChoice,
                                              beta1=0.9, beta2=0.999, attackid=attackID)




                attackAlgo.runForInhibition(attackID)
                attackAlgo.closeLogger()
                if attackAlgo._percentageTraining > 0.8:
                    break

            attackID = attackID + 1

            del fmodel
            del batcher
            del attackAlgo