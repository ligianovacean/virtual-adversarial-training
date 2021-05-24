import torch
import torchvision
import numpy as np


def write_images(X, r_vadv, writer):
    image_size = int(np.sqrt(X.shape[1]))
    img_grid = torchvision.utils.make_grid(torch.reshape(X[0:4, ...], [4, 1, image_size, image_size]),  padding=8, pad_value=1.0)
    # matplotlib_imshow(img_grid, one_channel=True)
    writer.add_image('images/original', img_grid)

    img_grid = torchvision.utils.make_grid(torch.reshape((X+r_vadv)[0:4, ...], [4, 1, image_size, image_size]), padding=8, pad_value=1.0)
    # matplotlib_imshow(img_grid, one_channel=True)
    writer.add_image('images/altered_images', img_grid)


def store_metrics(writer, train_loss, lds_loss, ce_loss, test_loss, test_accuracy, iteration):
    writer.add_scalar('Loss/total_train_loss', train_loss, iteration)
    writer.add_scalar('Loss/lds_train_loss', lds_loss, iteration)
    writer.add_scalar('Loss/ce_train_loss', ce_loss, iteration)
    writer.add_scalar('Loss/ce_valid_loss',test_loss, iteration)
    writer.add_scalar('Accuracy/valid_accuracy', test_accuracy, iteration)