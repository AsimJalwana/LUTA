
class BatchItem():

    def __init__(self):
        self._image = None
        self._name = None
        self._gradient = None
        self._targetLabel = None
        self._label = None
        self._ilabel = None
        self._alreadyHasTargetLabel = False

    @property
    def hasPredefinedTargetLabel(self):
        return self._alreadyHasTargetLabel

    @hasPredefinedTargetLabel.setter
    def hasPredefinedTargetLabel(self, hasLabel):
        self._alreadyHasTargetLabel = hasLabel

    @property
    def interimLabel(self):
        return self._ilabel

    @interimLabel.setter
    def interimLabel(self, value):
        self._ilabel = value

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, imageArray):
        self._image = imageArray

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, nameValue):
        self._name = nameValue

    @property
    def gradient(self):
        return self._gradient

    @gradient.setter
    def gradient(self, gradientValue):
        self._gradient = gradientValue

    @property
    def targetLabel(self):
        return self._targetLabel

    @targetLabel.setter
    def targetLabel(self, targetLabelValue):
        self._targetLabel = targetLabelValue

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, labelValue):
        self._label = labelValue

