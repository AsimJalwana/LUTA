from keras.applications.resnet50 import ResNet50 as resnet50
from keras.applications.resnet50 import decode_predictions as decode_predictions_resnet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

from keras.applications.vgg16 import VGG16 as vgg16
from keras.applications.vgg16 import decode_predictions as decode_predictions_vgg16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16

from keras.applications.inception_v3 import InceptionV3 as inceptionv3
from keras.applications.inception_v3 import decode_predictions as decode_predictions_inceptionv3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3

from keras.applications.mobilenet_v2 import MobileNetV2 as mobileNetV2
from keras.applications.mobilenet_v2 import decode_predictions as decode_predictions_mobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobileNetV2

from keras.backend.tensorflow_backend import set_session
from keras import backend as K

from ImageNetUtilities.ImageNetClasses import Decoder as DecoderImageNet

if 'tensorflow' == K.backend():
    import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

class AttackModels():
    # Do not change existing values.
    VGG16_MODEL = 1
    RESNET50_MODEL = 2
    INCEPTIONV3_MODEL = 5
    MOBILENETV2_MODEL = 6

    _availableModels = {VGG16_MODEL: 'VGG16', RESNET50_MODEL: 'ResNet50', INCEPTIONV3_MODEL: 'InceptionV3', MOBILENETV2_MODEL : 'MobileNetV2'}
    _availableAlgorithms = {3: "LInfinityBounded", 4: "L2Bounded"}

    @staticmethod
    def getModelDescription(model):
        return AttackModels._availableModels[model]

    @staticmethod
    def getAlgorithmDescription(algorithm):
        return AttackModels._availableAlgorithms[algorithm]

    @staticmethod
    def getModel(MODEL_CHOICE):
        # targetSize = (224, 224)
        model = None

        isCaffeModel = False
        if MODEL_CHOICE == AttackModels.VGG16_MODEL:
            model = vgg16(weights='imagenet')
            decode_predictions = decode_predictions_vgg16
            preprocess_input = preprocess_input_vgg16
            isCaffeModel = True

        if MODEL_CHOICE == AttackModels.RESNET50_MODEL:
            model = resnet50(weights='imagenet')
            decode_predictions = decode_predictions_resnet50
            preprocess_input = preprocess_input_resnet50
            isCaffeModel = True

        if MODEL_CHOICE == AttackModels.INCEPTIONV3_MODEL:
            model = inceptionv3(weights='imagenet')
            decode_predictions = decode_predictions_inceptionv3
            preprocess_input = preprocess_input_inceptionv3
            # targetSize = (299, 299)

        if MODEL_CHOICE == AttackModels.MOBILENETV2_MODEL:
            model = mobileNetV2(weights='imagenet')
            decode_predictions = decode_predictions_mobileNetV2
            preprocess_input = preprocess_input_mobileNetV2

        targetSize = AttackModels.getModelInputSize(MODEL_CHOICE)
        return model, targetSize, isCaffeModel, decode_predictions, preprocess_input

    @staticmethod
    def getModelInputSize(MODEL_CHOICE):
        targetSize = (224, 224)

        if MODEL_CHOICE == AttackModels.INCEPTIONV3_MODEL:
            targetSize = (299, 299)

        return targetSize

    @staticmethod
    def getModelDecoder(MODEL_CHOICE):
            return DecoderImageNet