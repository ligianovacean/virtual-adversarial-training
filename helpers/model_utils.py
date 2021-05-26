
from models.fcn_fnn_mnist_supervised import MNISTSupervisedFNN
from models.fcn_fnn_mnist_semisupervised import MNISTSemiSupervisedFNN
from models.cnn_synthetic import CNNSynthetic
from models.fcn_synthetic import FCNSynthetic
from models.cnn_mnist import LeNet5

import torch
import torch.optim as optim


def get_model(model_name):
    """Function that, based on the model name, instantiates a model object"""

    if model_name == 'mnist_supervised_fnn':
        return MNISTSupervisedFNN()
    elif model_name == 'mnist_semisupervised_fnn':
        return MNISTSemiSupervisedFNN()
    elif model_name == 'mnist_cnn':
        return LeNet5()
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


def get_optimizer(model, optimizer_name, lr, momentum):
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr)
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr, momentum=momentum)
    else:
        return None


def get_lr_scheduler(optimizer, scheduler_name, params):
    """ Returns the learning rate scheduler as defined by the scheduler name. Default: StepLR """
    if scheduler_name == 'exponential_decay':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=params[0])
    elif scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, gamma=params[0], step_size=params[1])
    else:
        return None


def save_model(path, model, iteration, optimizer, lr_scheduler):
    torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_Scheduler_state_dict': lr_scheduler.state_dict()
            }, path)
