import configparser
import os
import pathlib

class Manager():
    __PATH_SECTION = 'Paths'
    __VARIABLES_SECTION = 'Variables'

    def __init__(self):
        self.config = configparser.ConfigParser()
        if len(self.config.read( os.path.join( str(pathlib.Path(__file__).parent.parent), 'config.ini'))) == 0:
            print("Error!", "No Configuration file found! \n Please prepare config.ini")
        else:
            self.parsePathSection()


    def parsePathSection(self):
        if self.__class__.__PATH_SECTION in self.config:
            print("Paths section found in configuration file!")

    def getSavePath(self):
        return os.path.normpath(self.config[self.__class__.__PATH_SECTION]['saveDir'])

    def getDatasetPath(self):
        return os.path.normpath(self.config[self.__class__.__PATH_SECTION]['datasetDir'])

    def getInhibitionDatasetPath(self):
        return os.path.normpath(self.config[self.__class__.__PATH_SECTION]['inhibitionDatsetDir'])

    def getImageNetValidationPath(self):
        return os.path.normpath(self.config[self.__class__.__PATH_SECTION]['imageNetValidationDir'])

    def getImageNetValidationGroundTruthFilePath(self):
        return os.path.normpath(self.config[self.__class__.__PATH_SECTION]['imageNet2012ValidationGroundTruthFile'])

    def getImageNetValidationLabelMapFile(self):
        return os.path.normpath(self.config[self.__class__.__PATH_SECTION]['imageNet2012LabelMapFile'])

    def getSourceIdentities(self):
        return self.config[self.__class__.__VARIABLES_SECTION]['sourceIdentities']

    def getTargetIdentities(self):
        return self.config[self.__class__.__VARIABLES_SECTION]['targetIdentities']

    def getAttackModels(self):
        return self.config[self.__class__.__VARIABLES_SECTION]['attackModels']

    def getEtas(self):
        return self.config[self.__class__.__VARIABLES_SECTION]['etas']

    def getAlgorithmIDs(self):
        return self.config[self.__class__.__VARIABLES_SECTION]['algorithmId']