import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import MeanIoU
from src.callbacks import DisplayPrediction  # import your callback class

def train_pipeline(X_train, y_train, X_val, y_val, model=None, epochs=50, batch_size=4, lr=2e-4):
    """
    Train a U-Net model on the BUSI dataset.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model: Pre-built model; if None, it must be built externally
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate for Adam optimizer

    Returns:
        model: Trained model
        history: Training history object
    """

    if model is None:
        raise ValueError("Please provide a compiled model.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", MeanIoU(num_classes=2)]
    )

    # Early stopping
    earlystopper = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    # Model checkpoint
    checkpointer = ModelCheckpoint(
        "baseline_busi_unet.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # Random sample for display callback
    idx = np.random.randint(0, X_train.shape[0])
    display_cb = DisplayPrediction(X_train[idx], y_train[idx])

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[earlystopper, checkpointer, display_cb],
        verbose=1
    )

    return model, history
