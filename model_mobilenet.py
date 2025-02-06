import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large


def get_mobilenet_backbone(input_tensor, backbone_name='mobilenetv2'):
    """
    Mengambil backbone MobileNet berdasarkan nama.
    :param input_tensor: Tensor input untuk model.
    :param backbone_name: Nama backbone ('mobilenetv1', 'mobilenetv2', 'mobilenetv3small', 'mobilenetv3large').
    :return: Backbone feature extractor
    """
    if backbone_name == 'mobilenetv1':
        base_model = MobileNet(input_tensor=input_tensor, weights='imagenet', include_top=False)
    elif backbone_name == 'mobilenetv2':
        base_model = MobileNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)
    elif backbone_name == 'mobilenetv3small':
        base_model = MobileNetV3Small(input_tensor=input_tensor, weights='imagenet', include_top=False)
    elif backbone_name == 'mobilenetv3large':
        base_model = MobileNetV3Large(input_tensor=input_tensor, weights='imagenet', include_top=False)
    else:
        raise ValueError("Backbone tidak didukung. Pilih dari 'mobilenetv1', 'mobilenetv2', 'mobilenetv3small', 'mobilenetv3large'.")

    # Ambil fitur dari layer terakhir convolution
    feature_extractor = base_model.output

    return feature_extractor


# Contoh penggunaan
input_shape = (256, 256, 3)
input_tensor = layers.Input(shape=input_shape)

backbone = get_mobilenet_backbone(input_tensor, backbone_name='mobilenetv2')

model = models.Model(inputs=input_tensor, outputs=backbone)
model.summary()