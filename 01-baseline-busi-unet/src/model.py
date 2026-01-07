from tensorflow.keras import layers, Model

def unet_nn(input_shape=(128,128,1), dropout_rate=0.15):
  """
  Build a U-Net model for image segmentation (baseline configuration).

  Args:
      input_shape (tuple): Shape of input images, e.g., (128, 128, 1)
      dropout_rate (float): Dropout rate after each encoder block

  Returns:
      model (tf.keras.Model): Compiled U-Net model
  """

  inputs = layers.Input(input_shape)

  # === Encoder ===
  def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    p = layers.MaxPooling2D(2)(x)
    p = layers.Dropout(dropout_rate)(p)
    return x, p

  c1, p1 = conv_block(inputs, 24)
  c2, p2 = conv_block(p1, 48)
  c3, p3 = conv_block(p2, 96)
  c4, p4 = conv_block(p3, 192)

  # === Bottleneck ===
  b = layers.Conv2D(384, 3, padding='same', activation='relu')(p4)
  b = layers.BatchNormalization()(b)
  b = layers.Conv2D(384, 3, padding='same', activation='relu')(b)
  b = layers.BatchNormalization()(b)

  # === Decoder ===
  def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = layers.concatenate([x, skip])
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x

  d1 = decoder_block(b, c4, 192)
  d2 = decoder_block(d1, c3, 96)
  d3 = decoder_block(d2, c2, 48)
  d4 = decoder_block(d3, c1, 24)

  outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)

  model = Model(inputs, outputs)
  model.summary()
  return model
