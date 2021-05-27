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


def get_transformations(args):
    transforms = []
    if args.padding_size > 0:
        transforms += [torchvision.transforms.Pad(args.padding_size)]

    transforms += [torchvision.transforms.ToTensor()]

    transforms += [torchvision.transforms.Normalize((args.normalize_mean,), (args.normalize_std,))]
        
    # if args.vectorize:
    transforms += [VectorizeTransform()]

    transform = torchvision.transforms.Compose(transforms)

    return transform


def get_dataset(args):
    """
    Function used to load the train and test artificial_datasets.

    If dataset_name represents a torchvision dataset, save it to dataset_path. 
    Otherwise, read the dataset from dataset_path. 
    """

    transforms = get_transformations(args)
    if args.dataset == 'mnist':
        train_set = torchvision.datasets.MNIST(root=args.dataset_path, train=True, download=True, transform=transforms)
        test_set = torchvision.datasets.MNIST(root=args.dataset_path, train=False, download=True, transform=transforms)
    elif args.dataset == 'circles':
        circles_dataset = pd.read_pickle(args.dataset_path)
        train_set = list(zip(circles_dataset[0][0][0], circles_dataset[0][0][1]))
        test_set = list(zip(circles_dataset[0][1][0], circles_dataset[0][1][1]))
    else:
        moons_dataset = pd.read_pickle(args.dataset_path)
        train_set = list(zip(moons_dataset[0][0][0], moons_dataset[0][0][1]))
        test_set = list(zip(moons_dataset[0][1][0], moons_dataset[0][1][1]))

    # Compute size of validation set - if the specified 'valid_items' is lower than 1.0, it is a percentage, otherwise it is an absolute number
    valid_dataset = None
    if args.valid_split_items <= 1.0:
        valid_dataset_size = int(len(train_set) * args.valid_split_items)
    else:
        valid_dataset_size = int(args.valid_split_items)
    assert valid_dataset_size > 0 and valid_dataset_size < len(train_set), "Number of items in the validation set must be in range [0-train_set_size]."

    if args.labeled_items <= 1.0:
        labeled_dataset_size = int((len(train_set) - valid_dataset_size) * args.labeled_items)
    else:
        labeled_dataset_size = int(args.labeled_items)
    unlabeled_dataset_size = len(train_set) - valid_dataset_size

    unlabeled_dataset, valid_dataset = torch.utils.data.random_split(train_set, \
        [unlabeled_dataset_size, valid_dataset_size])
    labeled_dataset, _ = torch.utils.data.random_split(unlabeled_dataset, 
        [labeled_dataset_size, unlabeled_dataset_size - labeled_dataset_size])

    if not args.is_vat:
        unlabeled_dataset = None
    
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