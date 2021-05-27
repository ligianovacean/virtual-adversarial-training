import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torchvision
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from helpers.dataset_utils import VectorizeTransform
from helpers.dataset_utils import repeater, get_dataset, get_dataset_loaders
from helpers.plot_utils import plot_losses, plot_confusion_matrix, print_progress_bar
from helpers.model_utils import get_model, get_optimizer, get_lr_scheduler, save_model
from helpers.log_utils import store_metrics

from vat.VATLoss import VATLoss


# Workaround for MNIST download issues
new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
torchvision.datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in torchvision.datasets.MNIST.resources
]


def parse_args():
    parser = argparse.ArgumentParser(description="Semi-supervised learning experiments")

    # Dataset related parameters
    parser.add_argument('--dataset', default='mnist', required=True, \
                        help='Dataset name')
    parser.add_argument('--dataset_path', default=None, required=True, \
                        help='Path to dataset')
    parser.add_argument('--valid_split_items', type=float, default=0.0, \
                        help='Number of validation instances (default: 0)')
    parser.add_argument('--labeled_items', type=float, default=0.0, \
                        help='Percentage of labels to keep (default: 0.0)')

    # Model related parameters
    parser.add_argument('--model', default='lenet-5', \
                        help='Name of the model to train')
    parser.add_argument('--num_classes', type=int, default=None, \
                        help='Number of classes (default: model #classes)')
    parser.add_argument('--img_size', type=int, default=None, \
                        help='Image size (default: model default)')
    parser.add_argument('--unlabeled_batch_size', type=int, default=250, \
                        help='Unlabeled data loader batch size (default=250)')
    parser.add_argument('--labeled_batch_size', type=int, default=100, \
                    help='Labeled data loader batch size (default=100)')
    parser.add_argument('--test_batch_size', type=int, default=100, \
                        help='Labeled data loader batch size (default=100)')
    parser.add_argument('--num_workers', default=4, \
                        help='Number of workers (Default: 4)')

    # VAT related parameters
    parser.add_argument('--is_vat', action='store_true', default=False, \
                        help='Train using Virtual Adversarial Training')
    parser.add_argument('--reg_lambda', type=float, default=1.0, \
                        help='LDS regularization rate (default: 1.0)')
    parser.add_argument('--epsilon', type=float, default=1.0, \
                        help='VAT epsilon (default: 1.0)')
    parser.add_argument('--ip', type=int, default=1, \
                        help='Number of iterations in VAT\'s power iteration method (default: 1)')
    parser.add_argument('--xi', type=float, default=1e-6, \
                        help='VAT xi - perturbation scale (default: 1e-6)')

    # Data transformation related parameters
    parser.add_argument('--padding_size', type=int, default=-1, 
                        help='Pad image with specified padding if non-negative (default = -1)')
    parser.add_argument('--normalize_mean', type=float, default=0.1307, \
                        help='Normalization mean (default: 0.1307)')
    parser.add_argument('--normalize_std', type=float, default=0.3081, \
                        help='Normalization std (default: 0.3081)')

    # Model optim parameters
    parser.add_argument('--iterations', type=int, default=10000, \
                    help='Number of iterations to train (default: 100)')
    parser.add_argument('--optimizer', default='adam', \
                        help='Optimizer (default: adam)')
    parser.add_argument('--lr', type=float, default=1e-2, \
                        help='Learning rate (default: 1e-2')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--lr_scheduler', default='step', \
                        help='Learning rate scheduler (default: step)')
    parser.add_argument('--lr_decay', type=float, default=0.9, \
                        help='Learning rate decay (default: 0.9)')
    parser.add_argument('--lr_step_size', type=int, default=500, \
                        help='Nr. iterations after which the learning rate is decayed(default: 500)')
    parser.add_argument('--patience-epochs', type=int, default=10, \
                        help='Early stopping number of patience epochs (default: 10)')

    # Utility arguments
    parser.add_argument('--debug_mode', action='store_true', default=False, \
                        help='Whether to execute in debug mode (default: False)')
    parser.add_argument('--log_interval', type=int, default=500, \
                        help='How many batches to wait before logging training status (default: 500)')
    parser.add_argument('--experiment_name', default='experiment', 
                        help='Experiment name (default: experiment)')
    parser.add_argument('--repetitions', type=int, default=1,\
                        help='How many times to run the experiment (default: 1)')

    return parser.parse_args()


def print_results(iteration, total_train_loss, total_lds_loss, train_ce_loss, test_loss, accuracy, is_vat):
    print(f"Train iteration: {iteration+1}  \n\tTrain Loss: {np.around(total_train_loss, 6)}")
    if is_vat:
        print(f"\tLocal distribution smoothing: {np.around(total_lds_loss, 6)}")
    print(f"\tTrain CE Loss: {train_ce_loss}")

    print(f"\tValidation CE Loss: {np.around(test_loss, 6)}")
    print(f"\tValidation Acc: {np.around(accuracy, 2)}")


def train(model, optimizer, lr_scheduler, labeled_loader, unlabeled_loader, valid_loader, device, writer, experiments_folder, args):
    log_interval = args.log_interval
    iterations = args.iterations
    best_model_path = experiments_folder / "best_model"
    cpu = torch.device("cpu")

    # model.train()

    # Get infinite data loader
    labeled_loader = repeater(labeled_loader)
    if unlabeled_loader is not None:
        unlabeled_loader = repeater(unlabeled_loader)

    # Initialize training metrics
    total_train_loss = 0.0
    total_lds_loss = 0.0
    best_test_loss = float("inf")
    best_model = None

    for iteration in range(iterations):
        model.train()
        
        if labeled_loader is not None:
            labeled_batch = next(labeled_loader)
            labeled_data = labeled_batch[0].to(device)
            target = labeled_batch[1].to(device)

        if unlabeled_loader is not None:
            unlabeled_data = next(unlabeled_loader)
            unlabeled_data = unlabeled_data[0].to(device)
            
        optimizer.zero_grad()  # Set accumulated gradient to 0

        lds_loss = 0
        if args.is_vat:
            vat = VATLoss(args.epsilon, args.ip, args.xi)
            if unlabeled_loader is None:
                lds_loss = vat(labeled_data, model, device, writer)
            else:
                lds_loss = vat(unlabeled_data, model, device, writer)
            
            if labeled_data is not None:
                output = model(labeled_data)
                ce_loss = F.cross_entropy(output, target)
                loss = ce_loss + args.reg_lambda * lds_loss
            else:
                loss = args.reg_lambda * lds_loss
        else:
            output = model(labeled_data)
            loss = F.cross_entropy(output, target)

        loss.backward()  # Backward step
        optimizer.step()  # Update step
        lr_scheduler.step() # Learning rate update step
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], iteration)

        # Validation and logging 
        total_train_loss += loss.item()
        total_lds_loss += lds_loss
        if (iteration +1 ) % log_interval == 0:
            # Run inference
            test_loss, test_accuracy = inference(model, valid_loader, device)

            # Write metrics to TensorBoard
            train_loss = total_train_loss / log_interval
            lds_loss = total_lds_loss/log_interval
            ce_loss = train_loss - args.reg_lambda * lds_loss
            idx = (iteration + 1) // log_interval - 1
            store_metrics(writer, train_loss, lds_loss, ce_loss, test_loss, test_accuracy, iteration)
            
            # Save model if it outperforms previous best model
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = model
                save_model(best_model_path, model, iteration, optimizer, lr_scheduler)

            # Update progress bar
            print_progress_bar(iteration+1, iterations)

            total_train_loss = 0.0
            total_lds_loss = 0.0        

    return best_model


def inference(model, data_loader, device):
    model.eval()

    loss = 0.0
    count = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            count += 1

        loss /= count

    return loss / count, 100. * correct / len(data_loader.dataset)


def main(args, experiments_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging via TensorBoard
    writer = SummaryWriter(experiments_folder / "runs")

    labeled_dataset, unlabeled_dataset, valid_set, test_set = get_dataset(args)
    labeled_loader, unlabeled_loader, valid_loader, test_loader = get_dataset_loaders(labeled_dataset, unlabeled_dataset, valid_set, test_set, args)
    assert labeled_loader is not None and test_loader is not None, "Labeled train dataset and test dataset cannot be empty."

    model = get_model(args.model)
    assert model is not None, "Specified network name not supported."
    model.to(device)

    optimizer = get_optimizer(model, args.optimizer, args.lr, args.momentum)
    assert optimizer is not None, "Specified optimizer not supported."

    lr_scheduler = get_lr_scheduler(optimizer, args.lr_scheduler, [args.lr_decay, args.lr_step_size])
    assert lr_scheduler is not None, "Specified LR Scheduler not supported."

    if valid_loader is not None:
        best_model = train(model, optimizer, lr_scheduler,
            labeled_loader, unlabeled_loader, valid_loader, 
            device, writer, experiments_folder,
            args)
    else:
        best_model = train(model, optimizer, lr_scheduler,
            labeled_loader, unlabeled_loader, test_loader, 
            device, writer, experiments_folder,
            args)

    _, test_accuracy = inference(best_model, test_loader, device)

    writer.close()

    return test_accuracy


if __name__ == "__main__":
    args = parse_args()
    print(f"\nRunning with arguments: \n\t{args}\n")

    acc_arr = np.zeros(args.repetitions)
    for id_x in range(args.repetitions):
        print(f"\nExperiment repetition {id_x}")
        experiments_folder = Path.cwd() / "output" / args.experiment_name / f"iter_{id_x}"

        test_accuracy = main(args, experiments_folder)
        acc_arr[id_x] = test_accuracy

    # Save parameters
    with open(Path.cwd() / "output" / args.experiment_name / "parameters.txt", "w") as f:
        f.write(str(args))

    # Log summary metrics
    summary_writer = SummaryWriter(Path.cwd() / "output" / args.experiment_name / "summary")
    for id_x, acc in enumerate(acc_arr):
        summary_writer.add_scalar('Test accuracy', acc, id_x)
    summary_writer.add_scalar('Average accuracy', np.mean(acc_arr) - 1.96 * np.std(acc_arr) / np.sqrt(args.repetitions), 0)
    summary_writer.add_scalar('Average accuracy', np.mean(acc_arr), 1)
    summary_writer.add_scalar('Average accuracy', np.mean(acc_arr) + 1.96 * np.std(acc_arr) / np.sqrt(args.repetitions), 2)