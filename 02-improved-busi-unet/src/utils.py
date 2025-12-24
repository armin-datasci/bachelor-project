import os
import matplotlib.pyplot as plt

def smooth_curve(y, beta=0.9):
    smoothed = []
    v = 0
    for i, val in enumerate(y):
        v = beta*v + (1-beta)*val
        smoothed.append(v / (1-beta**(i+1)))
    return smoothed

def plot_history(history):
    keys = history.history.keys()
    miou_keys = [k for k in keys if "mean_io" in k]
    dice_keys = [k for k in keys if "soft_dice_coef" in k]

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(smooth_curve(history.history["loss"]), label="Train")
    plt.plot(smooth_curve(history.history["val_loss"]), label="Val")
    plt.title("Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(smooth_curve(history.history["accuracy"]), label="Train")
    plt.plot(smooth_curve(history.history["val_accuracy"]), label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

    if miou_keys:
        plt.figure()
        plt.plot(smooth_curve(history.history[miou_keys[0]]), label="Train Mean IoU")
        plt.plot(smooth_curve(history.history["val_" + miou_keys[0]]), label="Val Mean IoU")
        plt.title("Mean IoU")
        plt.legend()
        plt.show()

    if dice_keys:
        plt.figure()
        plt.plot(smooth_curve(history.history[dice_keys[0]]), label="Train Soft Dice")
        plt.plot(smooth_curve(history.history["val_" + dice_keys[0]]), label="Val Soft Dice")
        plt.title("Soft Dice")
        plt.legend()
        plt.show()

def save_history_figures(history, fig_folder):
    os.makedirs(fig_folder, exist_ok=True)
    keys = history.history.keys()
    miou_keys = [k for k in keys if "mean_io" in k]
    dice_keys = [k for k in keys if "soft_dice_coef" in k]

    # Loss & Accuracy
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.savefig(os.path.join(fig_folder, "loss_accuracy.png"))
    plt.close()

    if miou_keys:
        plt.figure()
        plt.plot(history.history[miou_keys[0]], label="Train Mean IoU")
        plt.plot(history.history["val_" + miou_keys[0]], label="Val Mean IoU")
        plt.title("Mean IoU")
        plt.legend()
        plt.savefig(os.path.join(fig_folder, "mean_iou.png"))
        plt.close()

    if dice_keys:
        plt.figure()
        plt.plot(history.history[dice_keys[0]], label="Train Soft Dice")
        plt.plot(history.history["val_" + dice_keys[0]], label="Val Soft Dice")
        plt.title("Soft Dice")
        plt.legend()
        plt.savefig(os.path.join(fig_folder, "soft_dice.png"))
        plt.close()
