from tensorflow.keras import layers, Model

def unet_nn(input_shape=(192, 192, 1), dropout_rate=0.1):
    """
    U-Net architecture for IMPROVED version.

    - Higher input resolution (192x192)
    - Wider feature maps
    - Lower dropout
    """

    inputs = layers.Input(input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        p = layers.MaxPooling2D(2)(x)
        p = layers.Dropout(dropout_rate)(p)
        return x, p

    # ===== Encoder =====
    c1, p1 = conv_block(inputs, 32)
    c2, p2 = conv_block(p1, 64)
    c3, p3 = conv_block(p2, 128)
    c4, p4 = conv_block(p3, 256)

    # ===== Bottleneck =====
    b = layers.Conv2D(512, 3, padding='same', activation='relu')(p4)
    b = layers.BatchNormalization()(b)
    b = layers.Conv2D(512, 3, padding='same', activation='relu')(b)
    b = layers.BatchNormalization()(b)

    # ===== Decoder =====
    def decoder_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = layers.concatenate([x, skip])
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        return x

    d1 = decoder_block(b, c4, 256)
    d2 = decoder_block(d1, c3, 128)
    d3 = decoder_block(d2, c2, 64)
    d4 = decoder_block(d3, c1, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)

    model = Model(inputs, outputs)
    model.summary()
    return model
