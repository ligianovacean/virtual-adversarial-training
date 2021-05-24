import matplotlib.pyplot as plt
import numpy as np
import torchvision


def plot_losses(losses):
    # Flatten loss plot - 10x1 averaging kernel 
    losses = np.convolve(losses, np.ones((10,)) / 10, mode='valid')
    plt.plot(losses)
    plt.grid()
    plt.title('Training loss')
    plt.show()


def plot_confusion_matrix(confusion_matrix):
    # normalized_confusion_matrix = confusion_matrix / len(testloader.dataset)
    normalized_confusion_matrix = confusion_matrix
    normalized_confusion_matrix[np.arange(10), np.arange(10)] -= normalized_confusion_matrix[
        np.arange(10), np.arange(10)]
    fig, ax = plt.subplots()

    im = ax.imshow(normalized_confusion_matrix, cmap='viridis')
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel('Truth')
    ax.set_ylabel('Predictions')

    fig.set_size_inches(8, 8, forward=True)
    plt.grid()

    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_title('Confusion matrix')
    fig.tight_layout()
    plt.show()


# Print iterations progress
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()