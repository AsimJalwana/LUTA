import logging
from datetime import datetime
import os

class Logger():
    DEBUG = logging.DEBUG
    INFO  = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

    def __init__(self, name, path, level=DEBUG):
        self._logger = None
        self._setLogger(name, path, level)


    def _setLogger(self, name, path, level):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        fileName = name + "-" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.log'

        fh = logging.FileHandler(os.path.join(path, fileName))
        fh.setLevel(level)

        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        self._logger.addHandler(ch)
        self._logger.addHandler(fh)

    @property
    def logger(self):
        return self._logger

    def close(self):
        handlers = self._logger.handlers[:]

        for handler in handlers:
            handler.close()
            self._logger.removeHandler(handler)
