import keras
import tensorflow as tf
from keras import Model, regularizers
from keras.src.applications.mobilenet_v3 import MobileNetV3Large
from keras.src.applications.mobilenet_v3 import preprocess_input
from keras.src.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, UpSampling2D, Concatenate, Conv2D
from keras.src.losses import binary_crossentropy
from keras.src.metrics import BinaryIoU
from keras.src.optimizers import Adam
from keras.src.metrics.f_score_metrics import F1Score
from tensorflow.keras import backend as K


def create_mobile_net_v3_large(img_size: tuple):
    mobile_net = MobileNetV3Large(
        input_shape=(*img_size, 3),
        weights='imagenet',
        include_top=False
    )
    mobile_net.trainable = False
    return mobile_net, preprocess_input


def create_classification_model(deep_model, num_classes:int, learning_rate:float=0.001) -> Model:
    assert num_classes > 0, "Число классов должно быть не меньше 1"
    if num_classes == 1:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
        metrics = ['accuracy', 'precision', 'recall']
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
        f1_score = F1Score(average='macro')
        metrics = ['accuracy', f1_score]

    x = deep_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = Dropout(0.4)(x)
    prediction = Dense(num_classes, activation=activation)(x)

    model = Model(inputs=deep_model.input, outputs=prediction)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    return model


def create_segmentation_model(deep_model,
                              is_leaf_seg: bool,
                              learning_rate: float = 0.001) -> Model:
    layer_names = [
        'expanded_conv_add',  # 128×128
        'expanded_conv_2_add',  # 64×64
        'expanded_conv_5_add',  # 32×32
        'expanded_conv_11_add',  # 16×16
    ]
    skip_connections = [deep_model.get_layer(name).output for name in layer_names]

    x = deep_model.output
    x = upsample_block(x, skip_connections[3], 256)  # 16×16
    x = upsample_block(x, skip_connections[2], 128)  # 32×32
    x = upsample_block(x, skip_connections[1], 64)  # 64×64
    x = upsample_block(x, skip_connections[0], 32)  # 128×128

    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    prediction = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=deep_model.input, outputs=prediction)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=bce_dice_loss if is_leaf_seg else tversky_loss,
        metrics=[
            BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou'),
            dice_coef
        ]
    )
    return model


def upsample_block(x, skip, filters):
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = Concatenate()([x, skip])
    x = Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    return x


def replace_class_head_and_compile(old_model:Model, num_classes:int, learning_rate:float=0.001) -> Model:
    x = old_model.layers[-2].output
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=old_model.input, outputs=predictions)

    f1_score = F1Score(average='macro')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', f1_score]
    )
    return model


@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred):
    smooth = 1.0

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


@keras.saving.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    bce = binary_crossentropy(y_true, y_pred)
    dsc = 1 - dice_coef(y_true, y_pred)
    return 0.5 * bce + 0.5 * dsc


@keras.saving.register_keras_serializable()
def tversky_loss(y_true, y_pred, ):
    alpha = 0.7
    beta = 0.3

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])

    tversky = (tp + 1e-7) / (tp + alpha * fn + beta * fp + 1e-7)
    return 1 - tf.reduce_mean(tversky)