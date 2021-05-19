import argparse
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torch
import torch.optim as optim
import torch.nn.functional as F

from models.fcn_mnist import MNISTModelFCN


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
    parser.add_argument('--dataset', default='mnist', required=True,\
        help='Dataset name')
    parser.add_argument('--dataset_path', default=None, required=True, \
        help='Path to dataset')
    parser.add_argument('--train_split', default='train', \
        help='Dataset train split (default: train)')
    parser.add_argument('--test_split', default='test', \
        help='Dataset test split (default: test)')
    
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
    
    # Data transformation related parameters


    # Optimizer parameters
    parser.add_argument('--optimizer', default='adam', \
        help='Optimizer (default: adam)')
    
    # Learning rate scheduler related parameters
    parser.add_argument('--lr_scheduler', default='step', \
        help='Learning rate scheduler (default: step)')
    parser.add_argument('--lr', type=float, default=1e-2, \
        help='Learning rate (default: 1e-2')    
    parser.add_argument('--epochs', type=int, default=100, \
        help='Number of epochs to train (default: 100)')
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
    losses = np.convolve(losses, np.ones((10,))/10, mode='valid')
    plt.plot(losses)
    plt.grid()
    plt.title('Training loss')
    plt.show()


def train(model, optimizer, train_loader, test_loader, device, log_interval=30, epochs=50):
    losses = np.zeros(len(train_loader) * epochs)

    model.train()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad() # Set accumulated gradient to 0

            output = model(data) # Forward step
            loss = F.nll_loss(output, target) # Negative log likelihood loss
            loss.backward() # Backward step
            optimizer.step() # Update step

            losses[len(train_loader) * epoch + batch_idx] = loss.item()
            if batch_idx % log_interval == 0:
                percentage = 100.0 * batch_idx / len(train_loader)
                print(f"Train epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({np.around(percentage, 2)}%)] \n\tTrain Loss: {np.around(loss.item(), 6)}")

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
    #normalized_confusion_matrix = confusion_matrix / len(testloader.dataset)
    normalized_confusion_matrix = confusion_matrix
    normalized_confusion_matrix[np.arange(10),np.arange(10)] -= normalized_confusion_matrix[np.arange(10),np.arange(10)]
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


def get_dataset(dataset_name, dataset_path, transforms,):
    """
    Function used to load the train and test datasets. 

    If dataset_name represents a torchvision dataset, save it to dataset_path. 
    Otherwise, read the dataset from dataset_path. 
    """

    if dataset_name == 'mnist':
        train_set = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transforms)
        test_set = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True, transform=transforms)
    else:
        return None, None

    return train_set, test_set


def get_dataset_loaders(train_set, test_set, batch_size, num_workers):
    """Function used to define the train and test dataset loaders (or generators) used to iterate the data."""

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader


def get_model(model_name):
    """Function that, based on the model name, instantiates a model object"""

    if model_name == 'mnist_fcn':
        return MNISTModelFCN()
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
    train_set, test_set = get_dataset(args.dataset, args.dataset_path, transforms)

    # if args.debug_mode:
    #     sample, gt = train_set[0]
    #     sample = np.array(sample)

    #     print(f'Sample shape is: {sample.shape}')
    #     print(f'Ground truth: {gt}\n')

    #     plt.imshow(sample.reshape(), cmap='gray', vmin=0, vmax=255)
    #     plt.show()

    train_loader, test_loader = get_dataset_loaders(train_set, test_set, args.batch_size, args.num_workers)

    model = get_model(args.model)
    model.to(device)

    optimizer = get_optimizer(model, args.optimizer, args.lr)

    log_interval = args.log_interval
    epochs = args.epochs

    train(model, optimizer, 
         train_loader, test_loader,
         device, 
         log_interval=log_interval, 
         epochs=epochs)

    evaluate(model, test_loader, optimizer, device)

    

if __name__ == "__main__":
    args = parse_args()

    main(args)
