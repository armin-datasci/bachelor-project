import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanIoU
from src.losses import bce_dice_loss, soft_dice_coef


def train_pipeline(
    X_train, y_train,
    X_val, y_val,
    model,
    epochs=50,
    batch_size=8,
    lr=2e-4,
    callbacks=None
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=bce_dice_loss,
        metrics=[
            'accuracy',
            MeanIoU(num_classes=2),
            soft_dice_coef
        ]
    )

    default_callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    if callbacks is not None:
        default_callbacks += callbacks

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=default_callbacks,
        verbose=1
    )

    return model, history
