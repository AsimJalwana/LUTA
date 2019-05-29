import glob
import os
import random

import PIL
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from AttackAlgorithms.AttackModels import AttackModels
from Batch.BatchItem import BatchItem


class Batcher():
    _pathToOtherClassesImage = None

    def __init__(self, targetSize, preProcessInput, pathOfImages, WnID, modelChoice, pathManager, pathOfTestingImages=None):
        print("WnId = ", WnID)
        self._batchSize = 64
        self._modelChoice = modelChoice
        self._currentBatchStart = -1
        self._currentBatchEnd = -1
        self._currentBatchImages = []
        self._allImagesItems = []
        self._allImagesItemsForTesting = []
        self._unperturbedOriginalImages = {}

        self._otherClassesImagesItemsForTraining = []

        self._pathOfImages = pathOfImages
        self._wnId = WnID
        self._fileNames = glob.glob( os.path.join(self._pathOfImages, "*.JPEG"))
        self._targetSize = targetSize

        self._preProcessInput = preProcessInput

        self._numberOfFiles = len(self._fileNames)
        self.db = None
        self._classificationModel = None
        self._imageNameIDMap = {}
        self._logger = None
        self._attackid = None
        self._pathOfImagesTesting = pathOfTestingImages
        self._fileNamesTesting = None

        if self._pathOfImagesTesting is not None:
            self._fileNamesTesting = glob.glob(os.path.join(self._pathOfImagesTesting, "*.JPEG"))

        Batcher._pathToOtherClassesImage = pathManager.getInhibitionDatasetPath()
        self._filesNamesOfOtherClasses = glob.glob(os.path.join(self.__class__._pathToOtherClassesImage, "*.JPEG"))
        self._enableInhibitionSamples = False

    @property
    def enableInhibitionSamples(self):
        return self._enableInhibitionSamples

    @enableInhibitionSamples.setter
    def enableInhibitionSamples(self, enable):
        self._enableInhibitionSamples = enable

    @property
    def preProcessInput(self):
        return self._preProcessInput

    @property
    def attackid(self):
        return self._attackId

    @attackid.setter
    def attackid(self, attackid):
        self._attackid = attackid

    @property
    def logger(self):
        return self.logger

    @logger.setter
    def logger(self, logger):
        self._logger = logger

    def getImageIDFromDescription(self, description):
        return self._imageNameIDMap[description]

    def getUnperturbedOriginalImage(self, fileName):
        return self._unperturbedOriginalImages[fileName]

    @property
    def databaseManager(self):
        return self.db

    @databaseManager.setter
    def databaseManager(self, dbManager):
        self.db = dbManager

    @property
    def classificationModel(self):
        return self._classificationModel

    @classificationModel.setter
    def classificationModel(self, classificationModel):
        self._classificationModel = classificationModel

    @property
    def WordNetID(self):
        return self._wnId

    @property
    def numberOfImages(self):
        return self._numberOfFiles

    @numberOfImages.setter
    def numberOfImages(self, value=-1):
        self._numberOfFiles = value

    @property
    def allImageItems(self):
        return self._allImagesItems

    @allImageItems.setter
    def allImageItems(self, value = []):
        self._allImagesItems = value

    @property
    def allImageItemsForTesting(self):
        return self._allImagesItemsForTesting

    @property
    def currentBatchImages(self):
        return self._currentBatchImages

    @currentBatchImages.setter
    def currentBatchImages(self, value=[]):
        self._currentBatchImages = value

    @property
    def targetSize(self):
        return self._targetSize

    @property
    def batchSize(self):
        return self._batchSize

    @property
    def prefixForPath(self):
        return self._pathOfImages

    @property
    def currentBatchStart(self):
        return self._currentBatchStart

    @currentBatchStart.setter
    def currentBatchStart(self, value=-1):
        self._currentBatchStart = value

    @property
    def currentBatchEnd(self):
        return self._currentBatchEnd

    @currentBatchEnd.setter
    def currentBatchEnd(self, value=-1):
        self._currentBatchEnd = value

    def loadOtherClassesImages(self, batchSize):
        self._logger.debug("BatchManager :: Loading random images from otherClasses Dataset")
        self._otherClassesImagesItemsForTraining = []

        randomSelectedFiles = random.sample(self._filesNamesOfOtherClasses, k=batchSize)

        for idx, file in enumerate(randomSelectedFiles):
            imageName = os.path.basename(file)
            wnid = imageName.split("_")[0]

            if wnid == self._wnId:
                continue

            imageOriginal = load_img(file)
            image = imageOriginal.resize(size=self.targetSize, resample=PIL.Image.BILINEAR)
            imageNumpy = img_to_array(image)

            batchItem = BatchItem()
            batchItem.image = imageNumpy
            batchItem.name = imageName
            batchItem.targetLabel = AttackModels.getModelDecoder(self._modelChoice).decodeWnIdToDeepModelId(wnid)
            batchItem.hasPredefinedTargetLabel = True

            self._otherClassesImagesItemsForTraining.append(batchItem)

        self._logger.debug("BatchManager :: length of other classes Images " + str(len(self._otherClassesImagesItemsForTraining)))

    def loadAllImages(self):
        self._logger.info("BatchManger::Started Loading all images")
        self.resetImageItems()
        self.db.add_wnidInfo(wnid=self.WordNetID, description=AttackModels.getModelDecoder(self._modelChoice).decodeWnIdToDescription(self.WordNetID))
        resolutionOfClassifier = str(self.targetSize[0]) + "," + str(self.targetSize[1])
        self.db.add_classifierInfo(classifierid=self.classificationModel, resolution=resolutionOfClassifier, description=AttackModels.getModelDescription(self.classificationModel))

        for file in self._fileNames:
            self.__addImageToDatabase(file=file, isTraining=1)

        if self._fileNamesTesting is not None:
            for file in self._fileNamesTesting:
                self.__addImageToDatabase(file, isTraining=0)

        self.databaseManager.commit()
        self.shuffleTheDataset()
        self._logger.info("BatchManger::All Images have been loaded, saved to database if necessary!")

    def __addImageToDatabase(self, file, isTraining):
        baseName = os.path.basename(file)
        imageId, imageDescription = self.db.doesImageInfoExists(description=baseName, wnid=self.WordNetID)

        imageOriginal = load_img(file)
        imageOriginalNumpy = img_to_array(imageOriginal, dtype=np.float32)
        sizeOfImage = np.shape(imageOriginalNumpy)
        resolutionString = str(sizeOfImage[0]) + "," + str(sizeOfImage[1]) + "," + str(sizeOfImage[2])

        if imageId is None:
            imageId = self.db.add_imageInfo(baseName, self.WordNetID, resolutionString)

        image = imageOriginal.resize(size=self.targetSize, resample=PIL.Image.BILINEAR)
        imageNumpy = img_to_array(image)

        self.db.add_trainingtesting_images(attackid=self._attackid, imageid=imageId, istraining=isTraining)

        inputImage = imageNumpy

        batchItem = BatchItem()
        batchItem.image = inputImage
        batchItem.name = baseName

        if isTraining == 1:
            self._allImagesItems.append(batchItem)
        else:
            self._allImagesItemsForTesting.append(batchItem)

        self._unperturbedOriginalImages[baseName] = np.copy(inputImage)
        self._imageNameIDMap[baseName] = imageId

    def resetImageItems(self):
        self._allImagesItems = []
        self._allImagesItemsForTesting = []
        self._unperturbedOriginalImages = {}
        self._imageNameIDMap = {}

    def shuffleTheDataset(self):
        random.shuffle(self._allImagesItems)

    def resetAllTest(self):
        for item in self._allImagesItemsForTesting:
            item.interimLabel = None
            item.image = np.copy(self._unperturbedOriginalImages[item.name])
            item.gradient = None

    def resetAll(self):
        for item in self._allImagesItems:
            item.interimLabel = None
            item.image = np.copy(self._unperturbedOriginalImages[item.name])
            item.gradient = None

    def getNextBatchOfItems(self):
        self._currentBatchImages = []

        if(self._currentBatchStart == -1):
            self._currentBatchStart = 0
        else:
            self._currentBatchStart = self._currentBatchEnd

        # always remember that last index is not inclusive!
        self._currentBatchEnd = self._currentBatchStart + self._batchSize

        if self._currentBatchEnd > self._numberOfFiles:
            self._currentBatchEnd = self._numberOfFiles

        if self._currentBatchStart > self._numberOfFiles - 1:
            return None

        self._currentBatchImages = self._allImagesItems[self._currentBatchStart:self._currentBatchEnd]

        if self._enableInhibitionSamples:
            self.loadOtherClassesImages(batchSize=64)
            self._currentBatchImages.extend(self._otherClassesImagesItemsForTraining)

        self._logger.debug("Size of current Batch of Images = " + str(len(self._currentBatchImages)))

        return self._currentBatchImages
