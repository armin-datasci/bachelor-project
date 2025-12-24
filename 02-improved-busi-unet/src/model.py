from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Conv2DTranspose, Concatenate, Dropout
)
from tensorflow.keras.models import Model

def build_unet(input_shape=(256,256,1), dropout_rate=0.1):
    inputs = Input(input_shape)
    x = inputs
    skips = []

    base_filters = [24, 48, 96, 192]

    # Encoder
    for f in base_filters:
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = MaxPooling2D(2)(x)
        x = Dropout(dropout_rate)(x)

    # Bottleneck
    x = Conv2D(384, 3, activation='relu', padding='same')(x)
    x = Conv2D(384, 3, activation='relu', padding='same')(x)

    # Decoder
    for i, f in enumerate(reversed(base_filters)):
        x = Conv2DTranspose(f, 2, strides=2, padding='same')(x)
        x = Concatenate()([x, skips[-(i + 1)]])
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Dropout(dropout_rate)(x)

    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs, outputs)
