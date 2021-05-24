import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

SYNTHETIC_DATASET_FILE = "syndata_1.pkl"
CIRCLES_DATASET_FILE = "syndata_2.pkl"


def display_testing_data_information(dataset):
    testing_samples = dataset[0][1]
    print("Number of training samples: %d" % len(testing_samples[0]))


def display_training_data_information(dataset):
    training_samples = dataset[0][1]
    print("Number of training samples: %d" % len(training_samples[0]))
    pca = PCA(2)
    data = pca.fit_transform(training_samples[0])
    data = pd.DataFrame(data)
    labels = training_samples[1]
    colormap = np.array(['r', 'b'])

    plt.scatter(data[0], data[1], s=100, c=colormap[labels])
    plt.show()


def display_dataset_information():
    dataset = pd.read_pickle(SYNTHETIC_DATASET_FILE)
    display_training_data_information(dataset)

    circles_dataset = pd.read_pickle(CIRCLES_DATASET_FILE)
    display_training_data_information(circles_dataset)


if __name__ == "__main__":
    print("Visualizing the synthetic dataset...")
    display_dataset_information()
    print("Finished visualizing the synthetic dataset!")
