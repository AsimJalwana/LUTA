import re
import numpy as np

class Parser():
    originalImageRegex = re.compile(r'target_class\:\s?(\d*)\s+target_class_string:\s?\"(\w*)\"', re.ASCII | re.IGNORECASE | re.VERBOSE)

    def __init__(self, pathToMapFile):
        self._path = pathToMapFile
        self._indexToIdMap = None
        self._idToIndexMap = None
        self._idToFileNames = None

    def parse(self):
        mapFile = None

        with open(self._path, "r") as file:
            mapFile = file.read()

        results = re.findall(Parser.originalImageRegex, mapFile)
        self._indexToIdMap =  dict((int(key), val) for (key, val) in results)
        revDct = dict((val, int(key)) for (key, val) in results)
        self._idToIndexMap = revDct

    @property
    def idIndexMap(self):
        return self._idToIndexMap

    @property
    def indexIdMap(self):
        return self._indexToIdMap

    @staticmethod
    def readFileInToNumpyArray(pathToFile):
        fileLines = None

        with open(pathToFile, "r") as file:
            fileLines = file.readlines()

        indicesArray = np.zeros((1, len(fileLines)))

        for index, line in enumerate(fileLines):
            indicesArray[0,index] = int(line)

        return indicesArray

    def getMapOfWnidToFileNames(self, pathOfGroundTruth):
        indicesArray = Parser.readFileInToNumpyArray(pathOfGroundTruth)

        if self._indexToIdMap is None:
            self.parse()

        indexToIdMap = self._indexToIdMap

        mapOfWnidToFileNames = dict()

        for i in range(0, np.shape(indicesArray)[1]):
            index = indicesArray[0, i]
            wnid = indexToIdMap[index]
            _, occurrence = np.where(indicesArray == index)
            occurrence = occurrence + 1
            occurenceFileNames = ['ILSVRC2012_val_' + str(index).zfill(8) + '.JPEG' for index in occurrence]
            mapOfWnidToFileNames[wnid] = occurenceFileNames

        self._idToFileNames = mapOfWnidToFileNames
        return self._idToFileNames