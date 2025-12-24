import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import numpy as np

class DisplayPrediction(Callback):
    """
    Keras callback to display a prediction after each epoch.
    """

    def __init__(self, image, mask):
        super().__init__()
        self.image = image
        self.mask = mask

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(
            np.expand_dims(self.image, axis=0), verbose=0
        )[0,:,:,0] > 0.5

        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.imshow(self.image[:,:,0], cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(self.mask[:,:,0], cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(pred, cmap='gray')
        plt.title(f"Prediction {epoch+1}")
        plt.axis("off")

        plt.show()
