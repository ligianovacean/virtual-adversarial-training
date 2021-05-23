import torchvision
import torch
import torch.nn.functional as F


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


def repeater(data_loader):
    while True:
        for data in data_loader:
            yield data


def get_standard_transformations():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        VectorizeTransform()
    ])

    return transform


def get_dataset(dataset_name, dataset_path, nr_labels_to_remove, valid_items):
    """
    Function used to load the train and test artificial_datasets.

    If dataset_name represents a torchvision dataset, save it to dataset_path. 
    Otherwise, read the dataset from dataset_path. 
    """

    if dataset_name == 'mnist':
        transforms = get_standard_transformations()
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

    # Compute size of validation set - if the specified 'valid_items' is lower than 1.0, it is a percentage, otherwise it is an absolute number
    valid_dataset = None
    if valid_items <= 1.0:
        valid_dataset_size = int(len(train_set) * valid_items)
    else:
        valid_dataset_size = int(valid_items)
    assert valid_dataset_size > 0 and valid_dataset_size < len(train_set), "Number of items in the validation set must be in range [0-train_set_size]."
    
    # Compute the size of the labeled and unlabeled sets - percentage vs. absolut value specification
    if nr_labels_to_remove <= 1.0:
        unlabeled_dataset_size = int((len(train_set) - valid_dataset_size) * nr_labels_to_remove)
    else:
        unlabeled_dataset_size = int(nr_labels_to_remove)
    labeled_dataset_size = len(train_set) - valid_dataset_size - unlabeled_dataset_size

    assert unlabeled_dataset_size >= -1 and unlabeled_dataset_size < len(train_set), "Number of unlabeled samples not in range [0-train_set_size]."

    labeled_dataset, unlabeled_dataset, valid_dataset = torch.utils.data.random_split(train_set, \
        [labeled_dataset_size, unlabeled_dataset_size, valid_dataset_size])

    return labeled_dataset, unlabeled_dataset, valid_dataset, test_set


def get_dataset_loaders(labeled_dataset, unlabeled_dataset, valid_set, test_set, args):
    """Function used to define the train and test dataset loaders (or generators) used to iterate the data."""

    unlabeled_loader = labeled_loader = validation_loader = None
    if unlabeled_dataset is not None and len(unlabeled_dataset) != 0:
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_dataset, shuffle=True, batch_size=args.unlabeled_batch_size, num_workers=args.num_workers)
    if labeled_dataset is not None and len(labeled_dataset) != 0:
        labeled_loader = torch.utils.data.DataLoader(labeled_dataset, shuffle=True, batch_size=args.labeled_batch_size, num_workers=args.num_workers)
    if valid_set is not None and len(valid_set) != 0:
        valid_loader = torch.utils.data.DataLoader(valid_set, shuffle=True, batch_size=args.test_batch_size, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=args.test_batch_size, num_workers=args.num_workers)

    return labeled_loader, unlabeled_loader, valid_loader, test_loader