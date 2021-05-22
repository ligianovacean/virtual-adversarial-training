import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torchvision
import torch
import torch.optim as optim
import torch.nn.functional as F

from models.fcn_mnist import MNISTModelFCN
from models.cnn_synthetic import CNNSynthetic
from models.fcn_synthetic import FCNSynthetic

from vat.VATLoss import VATLoss

# Workaround for MNIST download issues
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
torchvision.datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in torchvision.datasets.MNIST.resources
]


class VectorizeTransform:
    """Class constructor. Does nothing.
    """

    def __init__(self):
        pass

    """Implementation of operator()(sample).

    :param sample: Input image.
    :return: Vectorized (flattened) image.
    """

    def __call__(self, sample):
        return sample.view(-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Semi-supervised learning experiments")

    # Dataset related parameters
    parser.add_argument('--dataset', default='mnist', required=True, \
                        help='Dataset name')
    parser.add_argument('--dataset_path', default=None, required=True, \
                        help='Path to dataset')
    parser.add_argument('--train_split', default='train', \
                        help='Dataset train split (default: train)')
    parser.add_argument('--test_split', default='test', \
                        help='Dataset test split (default: test)')
    parser.add_argument('--unlabeled_percentage', type=int, default=0, \
                        help='Percentage of labels to remove')

    # Model related parameters
    parser.add_argument('--model', default='lenet-5', \
                        help='Name of the model to train')
    parser.add_argument('--num_classes', type=int, default=None, \
                        help='Number of classes (default: model #classes)')
    parser.add_argument('--img_size', type=int, default=None, \
                        help='Image size (default: model default)')
    parser.add_argument('--batch_size', type=int, default=64, \
                        help='Batch size (default=64)')
    parser.add_argument('--num_workers', default=2, \
                        help='Number of workers (Default: 2)')
    parser.add_argument('--is_vat', action='store_true', default=False, \
                        help='Train using Virtual Adversarial Training')
    parser.add_argument('--reg_lambda', type=float, default=1.0, \
                        help='LDS regularization rate')
    # Data transformation related parameters

    # Optimizer parameters
    parser.add_argument('--optimizer', default='adam', \
                        help='Optimizer (default: adam)')

    # Learning rate scheduler related parameters
    parser.add_argument('--lr_scheduler', default='step', \
                        help='Learning rate scheduler (default: step)')
    parser.add_argument('--lr', type=float, default=1e-2, \
                        help='Learning rate (default: 1e-2')
    parser.add_argument('--iterations', type=int, default=10000, \
                        help='Number of iterations to train (default: 100)')
    parser.add_argument('--patience-epochs', type=int, default=10, \
                        help='Patience epochs (default: 10)')

    # Utility arguments
    parser.add_argument('--debug_mode', action='store_true', default=False, \
                        help='Whether to execute in debug mode (default: False)')
    parser.add_argument('--log_interval', type=int, default=10, \
                        help='How many batches to wait before logging training status (default: 10)')

    return parser.parse_args()


def plot_losses(losses):
    # Flatten loss plot - 10x1 averaging kernel 
    losses = np.convolve(losses, np.ones((10,)) / 10, mode='valid')
    plt.plot(losses)
    plt.grid()
    plt.title('Training loss')
    plt.show()


def repeater(data_loader):
    while True:
        for data in data_loader:
            yield data


def train(model, optimizer, labeled_loader, unlabeled_loader, test_loader, device, log_interval=30, iterations=50, is_vat=False,
          reg_lambda=1.0):
    losses = np.zeros(iterations)

    model.train()

    labeled_loader = repeater(labeled_loader)
    unlabeled_loader = repeater(unlabeled_loader)

    for iteration in range(iterations):
        labeled_batch = next(labeled_loader)
        unlabeled_data = next(unlabeled_loader)
        unlabeled_data = unlabeled_data[0].to(device)

        labeled_data = labeled_batch[0].to(device)
        target = labeled_batch[1].to(device)
        optimizer.zero_grad()  # Set accumulated gradient to 0

        lds_loss = 0
        if is_vat:
            vat = VATLoss()
            lds_loss = vat(unlabeled_data, model, device)
            output = model(labeled_data)
            ce_loss = F.cross_entropy(output, target)
            loss = ce_loss + reg_lambda * lds_loss
        else:
            output = model(labeled_data)
            loss = F.cross_entropy(output, target)
        loss.backward()  # Backward step
        optimizer.step()  # Update step

        losses[iteration] = loss.item()
        if iteration % log_interval == 0:
            print(f"Train iteration: {iteration}  \n\tTrain Loss: {np.around(loss.item(), 6)}")
            print(f"Local distribution smoothing: {lds_loss}")

    plot_losses(losses)


def evaluate(model, test_loader, optimizer, device):
    model.eval()
    test_loss = 0.0
    correct = 0

    confusion_matrix = np.zeros((10, 10))

    cpu = torch.device("cpu")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            confusion_matrix[pred.cpu()[:, 0], target.cpu()] += 1

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    plot_confusion_matrix(confusion_matrix)


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


def get_transformations(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        VectorizeTransform()
    ])

    return transform


def get_dataset(dataset_name, dataset_path, transforms, percentage_of_labels_to_remove):
    """
    Function used to load the train and test artificial_datasets.

    If dataset_name represents a torchvision dataset, save it to dataset_path. 
    Otherwise, read the dataset from dataset_path. 
    """

    if dataset_name == 'mnist':
        train_set = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transforms)
        test_set = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True, transform=transforms)
    elif dataset_name == 'circles':
        circles_dataset = pd.read_pickle(dataset_path)
        train_set = list(zip(circles_dataset[0][0][0], circles_dataset[0][0][1]))
        test_set = list(zip(circles_dataset[0][1][0], circles_dataset[0][1][1]))
    else:
        moons_dataset = pd.read_pickle(dataset_path)
        train_set = list(zip(moons_dataset[0][0][0], moons_dataset[0][0][1]))
        test_set = list(zip(moons_dataset[0][1][0], moons_dataset[0][1][1]))

    unlabeled_dataset_size = int(len(train_set) * percentage_of_labels_to_remove / 100.0)
    labeled_dataset_size = len(train_set) - unlabeled_dataset_size
    labeled_dataset, unlabeled_dataset = torch.utils.data.random_split(train_set, [unlabeled_dataset_size, labeled_dataset_size])

    return labeled_dataset, unlabeled_dataset, test_set


def get_dataset_loaders(labeled_dataset, unlabeled_dataset, test_set, batch_size, num_workers):
    """Function used to define the train and test dataset loaders (or generators) used to iterate the data."""

    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    labeled_loader = torch.utils.data.DataLoader(labeled_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    return labeled_loader, unlabeled_loader, test_loader


def get_model(model_name):
    """Function that, based on the model name, instantiates a model object"""

    if model_name == 'mnist_fcn':
        return MNISTModelFCN()
    elif model_name == 'synthetic_fcn':
        fcn_network = FCNSynthetic()
        fcn_network.double()
        return fcn_network
    elif model_name == 'synthetic_cnn':
        cnn_network = CNNSynthetic()
        cnn_network.double()
        return cnn_network
    else:
        return None


def get_optimizer(model, optimizer_name, lr):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr)
    else:
        return None


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = get_transformations(args)
    labeled_dataset, unlabeled_dataset, test_set = get_dataset(args.dataset, args.dataset_path, transforms, args.unlabeled_percentage)

    # if args.debug_mode:
    #     sample, gt = train_set[0]
    #     sample = np.array(sample)

    #     print(f'Sample shape is: {sample.shape}')
    #     print(f'Ground truth: {gt}\n')

    #     plt.imshow(sample.reshape(), cmap='gray', vmin=0, vmax=255)
    #     plt.show()

    labeled_loader, unlabeled_loader, test_loader = get_dataset_loaders(labeled_dataset, unlabeled_dataset, test_set, args.batch_size, args.num_workers)

    model = get_model(args.model)
    model.to(device)

    optimizer = get_optimizer(model, args.optimizer, args.lr)

    log_interval = args.log_interval
    iterations = args.iterations

    train(model, optimizer,
          labeled_loader, unlabeled_loader, test_loader,
          device,
          log_interval=log_interval,
          iterations=iterations, is_vat=args.is_vat,
          reg_lambda=args.reg_lambda)

    evaluate(model, test_loader, optimizer, device)


if __name__ == "__main__":
    args = parse_args()

    main(args)
