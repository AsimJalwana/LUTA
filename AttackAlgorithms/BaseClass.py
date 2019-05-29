import os
from abc import ABC, abstractmethod

import math
import numpy as np

from AttacksDatabaseManager import AttacksDBManager
from Logger.Logger import Logger
from AttackAlgorithms.AttackModels import AttackModels


class AlgorithmsBaseClass(ABC):
    _algorithmID = -1
    _algorithmDescription = "Please define"
    _trainingAccuracyThreshold = 0.80
    _NUMBER_OF_CLASSES = -1
    _NUMBER_OF_ITERATION_IN_A_RUN = 100
    _NUMBER_OF_ITERATION_IN_INHIBITION = 100
    _SAVE_RESULTS_ITERATION = 50
    _DECODER = None

    def __init__(self, fmodel, batcher, targetLabel, savePath, eta, modelChoice, isCaffeModel, beta1=0, beta2=0,
                 attackid=None):
        super().__init__()
        AlgorithmsBaseClass._NUMBER_OF_CLASSES = 1000
        self.fmodel = fmodel
        self.batcher = batcher
        self.targetLabel = targetLabel
        self.savePath = savePath
        self.eta = eta
        self.perturbation = None
        self.upsilon = None
        self.omega = None
        self.beta1 = beta1
        self.beta2 = beta2
        self.attackid = attackid
        self.isCaffeModel = isCaffeModel
        self.doNotRunTheAttack = False
        self.modelChoice = modelChoice
        self.setLogger()
        self.setDBManager()
        self.logHyperparametersDetails()
        self.histogramOfTrainingPredictions = None
        self.histogramOfTestingPredictions = None
        self.normRatioOfSourceClassToOtherClass = 1

        self._percentageTraining = 0
        self._percentageTesting = 0
        self._runningIterationForContinousRuns = 1

    @staticmethod
    def setNumberOfIterationsInARun(iterations):
        AlgorithmsBaseClass._NUMBER_OF_ITERATION_IN_A_RUN = iterations

    @property
    def runningIteration(self):
        return self._runningIterationForContinousRuns

    @property
    def percentageTrainingAccuracy(self):
        return self._percentageTraining

    @property
    def percentageTestingAccuracy(self):
        return self._percentageTesting

    def setDBManager(self):
        self.db = AttacksDBManager.GradientBasedAttacksDatabase.connect(
            os.path.join(self.savePath, "AdversarialAttackResults.db"), self.loggerHandle)
        self.db.create_tables()
        self.db.add_attackalgorithmInfo(algorithmid=AlgorithmsBaseClass._algorithmID,
                                        description=AlgorithmsBaseClass._algorithmDescription)
        self.batcher.databaseManager = self.db
        self.batcher.classificationModel = self.modelChoice
        self.batcher.logger = self.loggerHandle

        self.iteration = 1
        self.epoch = 1
        self.perturbationFromDB = None
        self.upsilonFromDB = None
        self.omegaFromDB = None
        self.numberOfBatchesTobeSkipped = 0

        if self.attackid is None:
            self.attackid = self.db.add_attackInfo(attackid=None, description=AlgorithmsBaseClass._algorithmDescription,
                                                   classifierid=self.modelChoice)
            self.db.add_hyperparameterInfo(attackid=self.attackid, eta=self.eta, beta1=self.beta1,
                                                beta2=self.beta2,
                                                batchsize=self.batcher.batchSize,
                                                algorithmid=AlgorithmsBaseClass._algorithmID,
                                                groundlabel=AttackModels.getModelDecoder(self.modelChoice).decodeWnIdToDeepModelId(self.batcher.WordNetID),
                                                targetlabel=self.targetLabel,
                                                otherParameters='')
        else:
            self.loadFromSavedResults()
            self.db.clearAttackTestingTrainingPerformance(self.attackid, self.iteration)
            self.verifyHyperParameters()

        self.batcher.attackid = self.attackid

    def loadFromSavedResults(self):
        self.loggerHandle.info("As attackID has been passed, so loading from saved results..")
        attackid, iteration, epoch, perturbedimageRaw, upsilonimageRaw, omegaimageRaw = self.db.findDataFromLastRun(
            self.attackid)

        if iteration is None:
            iteration = 0
            epoch = 1
            perturbedimageRaw = np.zeros((self.batcher.targetSize[0], self.batcher.targetSize[1], 3), dtype=np.float32)
            upsilonimageRaw = np.zeros((self.batcher.targetSize[0], self.batcher.targetSize[1], 3), dtype=np.float32)
            omegaimageRaw = np.zeros((self.batcher.targetSize[0], self.batcher.targetSize[1], 3), dtype=np.float32)

        self.iteration = iteration + 1
        self.epoch = epoch

        self.perturbationFromDB = np.copy(perturbedimageRaw)
        self.upsilonFromDB = np.copy(upsilonimageRaw)
        self.omegaFromDB = np.copy(omegaimageRaw)

        self.loggerHandle.info(
            "AttackID = " + str(attackid) + ", iteration = " + str(iteration) + ", perturbedImageRawShape = " + str(
                np.shape(perturbedimageRaw)))

    def verifyHyperParameters(self):
        eta, beta1, beta2, batchsize, algorithmid, groundlabel, targetlabel = self.db.loadHyperparametersForTheAttack(
            self.attackid)

        if self.eta == eta and self.beta1 == beta1 and self.beta2 == beta2 and self.batcher.batchSize == batchsize and AlgorithmsBaseClass._algorithmID == algorithmid and AttackModels.getModelDecoder(self.modelChoice).decodeWnIdToDeepModelId(
                self.batcher.WordNetID) == groundlabel and self.targetLabel == targetlabel:
            self.loggerHandle.info("The hyperparameters match!")
            self.doNotRunTheAttack = False
        else:
            self.loggerHandle.error("The passed hyperparameters do not match the record")
            self.doNotRunTheAttack = True

    def logHyperparametersDetails(self):
        self.loggerHandle.info("***********************************************************")
        self.loggerHandle.info("WNID = " + self.batcher.WordNetID)
        self.loggerHandle.info("TargetLabel = " + str(self.targetLabel))
        self.loggerHandle.info("ETA = " + str(self.eta))
        self.loggerHandle.info("BETA1 = " + str(self.beta1))
        self.loggerHandle.info("BETA2 = " + str(self.beta2))
        self.loggerHandle.info("AttackID = " + str(self.attackid))
        self.loggerHandle.info("BatchSize = " + str(self.batcher.batchSize))
        self.loggerHandle.info("***********************************************************")

    def setLogger(self):
        self.logger = Logger(name='AttackResultsLogger', path=self.savePath, level=Logger.DEBUG)
        self.loggerHandle = self.logger.logger

    '''
        Updates the interim labels for as per current perturbation on untouched images!

    '''

    def updateLabelsOnly(self):
        for item in self.batcher.allImageItems:
            image = np.copy(self.batcher.getUnperturbedOriginalImage(item.name))
            imagePerturbed = image - self.perturbation
            np.clip(imagePerturbed, 0, 255, out=imagePerturbed)

            fmodelPredictions = self.fmodel.predictions(self.batcher.preProcessInput(imagePerturbed))
            predictedLabel = np.argmax(fmodelPredictions)
            self.histogramOfTrainingPredictions[0, predictedLabel] += 1

            if item.label is None:
                item.label = predictedLabel
                self.db.add_classifierImage(classifierid=self.modelChoice,
                                            imageid=self.batcher.getImageIDFromDescription(item.name),
                                            plabel=int(predictedLabel))
                self.db.commit()

            if item.targetLabel is None:
                item.targetLabel = self.targetLabel

            item.interimLabel = predictedLabel

    def addPeturbationToBatch(self, listOfBatchItems):
        for item in listOfBatchItems:
            perturbedImage = item.image - self.perturbation
            np.clip(perturbedImage, 0, 255, out=perturbedImage)
            item.image = perturbedImage

    def updateGradientsOnABatch(self, listOfBatchItems):
        gradientsMeanNormSourceClass = 0
        gradientsMeanNormOfOtherClass = 0

        numberOfSourceClasses = 0
        numberOfOtherClasses = 0

        for item in listOfBatchItems:

            if item.hasPredefinedTargetLabel:
                targetLabel = item.targetLabel
            else:
                targetLabel = self.targetLabel

            fmodelPredictions, gradient = self.fmodel.predictions_and_gradient(self.batcher.preProcessInput(item.image), label=targetLabel)

            if item.hasPredefinedTargetLabel:
                gradientsMeanNormOfOtherClass = gradientsMeanNormOfOtherClass + np.linalg.norm(gradient)
                numberOfOtherClasses += 1
            else:
                gradientsMeanNormSourceClass = gradientsMeanNormSourceClass + np.linalg.norm(gradient)
                numberOfSourceClasses += 1

            item.gradient = gradient
            if self.isCaffeModel:  # only for caffe based models
                item.gradient = gradient[:, :, ::-1]

            predictedLabel = np.argmax(fmodelPredictions)

            if item.label is None:
                item.label = predictedLabel

            if item.targetLabel is None:
                item.targetLabel = self.targetLabel

            item.interimLabel = predictedLabel

        if numberOfOtherClasses != 0:
            gradientsMeanNormSourceClass = gradientsMeanNormSourceClass / numberOfSourceClasses
            gradientsMeanNormOfOtherClass = gradientsMeanNormOfOtherClass / numberOfOtherClasses
            self.normRatioOfSourceClassToOtherClass = gradientsMeanNormSourceClass / gradientsMeanNormOfOtherClass
        else:
            self.normRatioOfSourceClassToOtherClass = None

    def findMeanOfSumOfGradients(self, listOfBatchItems):
        mean = np.zeros((self.batcher.targetSize[0], self.batcher.targetSize[1], 3), dtype=float)
        for item in listOfBatchItems:
            gradient = item.gradient

            if item.hasPredefinedTargetLabel:
                originalNorm = str(np.linalg.norm(gradient))
                gradient = gradient * self.normRatioOfSourceClassToOtherClass
                # self.loggerHandle.debug("Changing gradient norm from " + originalNorm + " to " + str(np.linalg.norm(gradient)))

            mean = mean + gradient
        mean = mean / len(listOfBatchItems)
        return mean

    def findPercentageOfCorrectlyPerturbedSamples(self):
        total = len(self.batcher.allImageItems)
        self.histogramOfTrainingPredictions = np.zeros((1, self.__class__._NUMBER_OF_CLASSES))
        self.updateLabelsOnly()
        success = 0
        for item in self.batcher.allImageItems:
            if item.interimLabel == item.targetLabel:
                success = success + 1
        return success / total, success

    def findTopMostTrainingPredictedClass(self):
        predictions = []

        for item in self.batcher.allImageItems:
            predictions.append(int(item.interimLabel))

        return self.most_frequent(predictions)

    def findTopMostTestingPredictionClass(self):
        predictions = []

        for item in self.batcher.allImageItemsForTesting:
            predictions.append(int(item.interimLabel))

        return self.most_frequent(predictions)

    def findTestingAccuracy(self):
        total = len(self.batcher.allImageItemsForTesting)
        self.histogramOfTestingPredictions = np.zeros((1, self.__class__._NUMBER_OF_CLASSES))
        self.updateLabelsForTestOnly()
        success = 0
        for item in self.batcher.allImageItemsForTesting:
            if item.interimLabel == item.targetLabel:
                success = success + 1
        return success / total, success

    def computeNormalizedXi(self, meanOfSumOfGradients):
        return meanOfSumOfGradients / np.linalg.norm(meanOfSumOfGradients)

    def updateAlgorithmVariables(self, listOfBatchItems, iteration, epochNumber):
        xi = self.findMeanOfSumOfGradients(listOfBatchItems)
        self.upsilon = self.beta1 * self.upsilon + (1 - self.beta1) * xi
        self.omega = self.beta2 * self.omega + (1 - self.beta2) * np.square(xi)

        upsilon = self.upsilon * math.sqrt(1 - pow(self.beta2, iteration))
        omega = np.sqrt(self.omega) * (1 - pow(self.beta1, iteration))
        incrementInPerturbation = upsilon / omega
        incrementInPerturbation = incrementInPerturbation / (np.linalg.norm(np.reshape(incrementInPerturbation, (-1, 1)), np.inf))
        # incrementInPerturbation = pow(0.99, epochNumber) * incrementInPerturbation
        incrementInPerturbation = 1.0 * incrementInPerturbation
        self.perturbation = self.perturbation + incrementInPerturbation

    @abstractmethod
    def updatePeturbationByBackProjection(self):
        pass

    def saveTheResults(self, i, epoch):
        perturbationImage = np.float32(np.copy(self.perturbation))
        upsilonImage = np.float32(np.copy(self.upsilon))
        omegaImage = np.float32(np.copy(self.omega))

        self.db.add_attack(attackid=self.attackid, iteration=i, epoch=epoch, perturbedImage=perturbationImage,
                                upsilonImage=upsilonImage, omegaImage=omegaImage)

    def initPerturbationImages(self):
        if self.perturbationFromDB is None:
            self.perturbation = np.zeros((self.batcher.targetSize[0], self.batcher.targetSize[1], 3), dtype=np.float32)
            self.upsilon = np.zeros((self.batcher.targetSize[0], self.batcher.targetSize[1], 3), dtype=np.float32)
            self.omega = np.zeros((self.batcher.targetSize[0], self.batcher.targetSize[1], 3), dtype=np.float32)
        else:
            self.perturbation = np.copy(self.perturbationFromDB)
            self.upsilon = np.copy(self.upsilonFromDB)
            self.omega = np.copy(self.omegaFromDB)

    def initPerturbationImagesFromOtherAttack(self, attackId):
        _, _, classifierid = self.db.get_attackinfo(attackId)
        targetSize = AttackModels.getModelInputSize(classifierid)
        _, _, _, self.perturbation, self.upsilon, self.omega = self.db.findDataFromLastRun(attackId, targetSize)

    def updateLabelsForTestOnly(self):
        for item in self.batcher.allImageItemsForTesting:
            image = np.copy(self.batcher.getUnperturbedOriginalImage(item.name))
            imagePerturbed = image - self.perturbation
            np.clip(imagePerturbed, 0, 255, out=imagePerturbed)

            fmodelPredictions = self.fmodel.predictions(self.batcher.preProcessInput(imagePerturbed))
            predictedLabel = np.argmax(fmodelPredictions)
            self.histogramOfTestingPredictions[0, predictedLabel] += 1

            if item.label is None:
                item.label = predictedLabel
                self.db.add_classifierImage(classifierid=self.modelChoice,
                                            imageid=self.batcher.getImageIDFromDescription(item.name),
                                            plabel=int(predictedLabel))
                self.db.commit()

            if item.targetLabel is None:
                item.targetLabel = self.targetLabel

            item.interimLabel = predictedLabel

    def setNumberOfBatchesToBeSkipped(self):
        numberOfTrainingSamples = len(self.batcher.allImageItems)
        numberOfIterationsInABatch = math.ceil(numberOfTrainingSamples / self.batcher.batchSize)
        startOfIterationInCurrentEpoch = (self.epoch - 1) * numberOfIterationsInABatch + 1
        self.numberOfBatchesTobeSkipped = self.iteration - startOfIterationInCurrentEpoch
        self.loggerHandle.info("Number of batches to be skipped = " + str(self.numberOfBatchesTobeSkipped))

    def most_frequent(self, List):
        return max(set(List), key=List.count)

    def findAccuraciesAndLog(self, i):
        percentage, numberOfCorrectlyPerturbedImages = self.findPercentageOfCorrectlyPerturbedSamples()
        percentageTesting, numberOfCorrectlyPerturbedTestImages = self.findTestingAccuracy()

        self.loggerHandle.info("............Fooling Ratio = " + ('%.5f' % percentage) + "  ... Total = " + str(
            numberOfCorrectlyPerturbedImages))

        self.loggerHandle.info(
            "............Test Success Ratio = " + ('%.5f' % percentageTesting) + "  ....... Total = " + str(
                numberOfCorrectlyPerturbedTestImages))

        mostPredictedTrainingClassLabel = self.findTopMostTrainingPredictedClass()
        mostPredictedTestingClassLabel = self.findTopMostTestingPredictionClass()

        self.loggerHandle.debug("Most Frequent Training Label :: = " + str(
            mostPredictedTrainingClassLabel) + ", description = " + AttackModels.getModelDecoder(self.modelChoice).decode(mostPredictedTrainingClassLabel))
        self.loggerHandle.debug("Most Frequent Testing  Label :: = " + str(
            mostPredictedTestingClassLabel) + ", description = " + AttackModels.getModelDecoder(self.modelChoice).decode(mostPredictedTestingClassLabel))

        self.db.add_attack_training_performance(self.attackid, i, percentage)
        self.db.add_attack_testing_performance(self.attackid, i, percentageTesting)
        self.db.add_attack_training_prediction(self.attackid, i, mostPredictedTrainingClassLabel)
        self.db.add_attack_testing_prediction(self.attackid, i, mostPredictedTestingClassLabel)
        self.db.add_attack_training_all_predictions(self.attackid, i, self.histogramOfTrainingPredictions)
        self.db.add_attack_testing_all_predictions(self.attackid, i, self.histogramOfTestingPredictions)

        return percentage, percentageTesting

    def runForInhibition(self, originalAttackId):
        if self.doNotRunTheAttack:
            self.loggerHandle.error("The flag doNotRunTheAttack is ON, terminating the experiment!")
            return

        self.batcher.enableInhibitionSamples = True
        self.batcher.loadAllImages()
        self.initPerturbationImagesFromOtherAttack(originalAttackId)
        epoch = self.epoch

        self.loggerHandle.info("The starting Iteration is " + str(self.iteration))

        if self.iteration != 1:
            # patch for case where step had started from 1!, to be removed in future
            self._percentageTraining, self._percentageTesting = self.findAccuraciesAndLog(i=self.iteration - 1)
            self.setNumberOfBatchesToBeSkipped()
            for i in range(0, self.numberOfBatchesTobeSkipped):
                _ = self.batcher.getNextBatchOfItems()
        else:
            self._percentageTraining, self._percentageTesting = self.findAccuraciesAndLog(i=0)

        start = self.iteration
        for self._runningIterationForContinousRuns in range(start, start + AlgorithmsBaseClass._NUMBER_OF_ITERATION_IN_INHIBITION):
            self.loggerHandle.info("Step = " + str(self._runningIterationForContinousRuns) + "....")
            listOfBatchItems = self.batcher.getNextBatchOfItems()

            if (listOfBatchItems is None):
                self.batcher.shuffleTheDataset()
                self.batcher.resetAll()
                self.loggerHandle.warning("NO data left, so reshuffling ............................................")
                self.batcher.currentBatchStart = -1
                self.batcher.currentBatchEnd = -1
                epoch = epoch + 1
                listOfBatchItems = self.batcher.getNextBatchOfItems()

            self.addPeturbationToBatch(listOfBatchItems)
            self.updateGradientsOnABatch(listOfBatchItems)
            self.updateAlgorithmVariables(listOfBatchItems, iteration=self._runningIterationForContinousRuns, epochNumber=epoch)
            self.updatePeturbationByBackProjection()

            self._percentageTraining, self._percentageTesting = self.findAccuraciesAndLog(self._runningIterationForContinousRuns)

            if self._runningIterationForContinousRuns % AlgorithmsBaseClass._SAVE_RESULTS_ITERATION == 0:
                self.loggerHandle.warning("Saving the iteration results... ")
                self.saveTheResults(self._runningIterationForContinousRuns, epoch)

            self.loggerHandle.debug("Perturbation = " + str(np.linalg.norm(self.perturbation)))
        self.logger.close()

    def run(self, checkForAccuracy):
        if self.doNotRunTheAttack:
            self.loggerHandle.error("The flag doNotRunTheAttack is ON, terminating the experiment!")
            return

        self.batcher.enableInhibitionSamples = False
        self.batcher.loadAllImages()
        self.initPerturbationImages()
        epoch = self.epoch

        self.loggerHandle.info("The starting Iteration is " + str(self.iteration))
        self._percentageTraining = 0
        self._percentageTesting = 0

        if self.iteration != 1:
            # patch for case where step had started from 1!, to be removed in future
            self._percentageTraining, self._percentageTesting = self.findAccuraciesAndLog(i=self.iteration - 1)
            if self._percentageTraining >= AlgorithmsBaseClass._trainingAccuracyThreshold:
                self.loggerHandle.info("The initial accuracy is greater than threshold, breaking!")
                return
            self.setNumberOfBatchesToBeSkipped()
            for i in range(0, self.numberOfBatchesTobeSkipped):
                _ = self.batcher.getNextBatchOfItems()
        else:
            self._percentageTraining, self._percentageTesting = self.findAccuraciesAndLog(i=0)
            if self._percentageTraining >= AlgorithmsBaseClass._trainingAccuracyThreshold:
                self.loggerHandle.info("The initial accuracy is greater than threshold, breaking!")
                return

        for self._runningIterationForContinousRuns in range(self.iteration, self.iteration + AlgorithmsBaseClass._NUMBER_OF_ITERATION_IN_A_RUN):
            self.loggerHandle.info("Step = " + str(self._runningIterationForContinousRuns) + "....")
            listOfBatchItems = self.batcher.getNextBatchOfItems()

            if (listOfBatchItems is None):
                self.batcher.shuffleTheDataset()
                self.batcher.resetAll()
                self.loggerHandle.warning(
                    "NO data left, so reshuffling ............................................")
                self.batcher.currentBatchStart = -1
                self.batcher.currentBatchEnd = -1
                epoch = epoch + 1
                listOfBatchItems = self.batcher.getNextBatchOfItems()

            self.addPeturbationToBatch(listOfBatchItems)
            self.updateGradientsOnABatch(listOfBatchItems)
            self.updateAlgorithmVariables(listOfBatchItems, iteration=self._runningIterationForContinousRuns, epochNumber=epoch)
            self.updatePeturbationByBackProjection()

            self._percentageTraining, self._percentageTesting = self.findAccuraciesAndLog(self._runningIterationForContinousRuns)

            if checkForAccuracy:
                if self._percentageTraining >= AlgorithmsBaseClass._trainingAccuracyThreshold:
                    self.loggerHandle.info("Achieved Accuracy so breaking....")
                    self.saveTheResults(self._runningIterationForContinousRuns, epoch)
                    break

            if self._runningIterationForContinousRuns % AlgorithmsBaseClass._SAVE_RESULTS_ITERATION == 0:
                self.loggerHandle.warning("Saving the iteration results... ")
                self.saveTheResults(self._runningIterationForContinousRuns, epoch)

            self.loggerHandle.debug("Perturbation = " + str(np.linalg.norm(self.perturbation)))

    def closeLogger(self):
        self.logger.close()