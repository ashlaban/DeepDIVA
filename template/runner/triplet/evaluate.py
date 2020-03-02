# Utils
import datetime
import json
import logging
import time
from sklearn.metrics import pairwise_distances_chunked
import numpy as np

# Torch related stuff
import torch
from tqdm import tqdm

# DeepDIVA
from util.evaluation.metrics import compute_mapk


def validate(val_loader, model, criterion, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to validate the model."""

    # This logs triplet loss information to the summary writer.
    _evaluate(val_loader, model, criterion, writer, epoch, 'val', no_cuda, log_interval, **kwargs)

    # mAP is the true metric, used for monitoring.
    return _evaluate_map(val_loader, model, writer, epoch, 'val', no_cuda, log_interval, **kwargs)


def test(test_loader, model, writer, epoch, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to test the model"""
    return _evaluate_map(test_loader, model, writer, epoch, 'test', no_cuda, log_interval, **kwargs)


def _evaluate(val_loader, model, criterion, writer, epoch, no_cuda, log_interval=25, **kwargs):
    """
    Evaluation routine

    Parameters
    ----------
    val_loader : torch.utils.data.DataLoader
        The dataloader of the train set.
    model : torch.nn.module
        The network model being used.
    criterion : torch.nn.loss
        The loss function used to compute the loss of the model.
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes).
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    Returns
    ----------
    int
        Placeholder 0. In the future this should become the FPR95
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Switch to train mode (turn off dropout & stuff)
    model.eval()

    # Turn on "triplet mode". Since we want to evaluate the triplet loss, we
    # need to enable this.
    val_loader.dataset.train = True
    val_loader.dataset.triplets = val_loader.dataset.generate_triplets()

    # Iterate over whole training set
    end = time.time()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (data_a, data_p, data_n) in pbar:

        if len(data_a.size()) == 5:
            bs, ncrops, c, h, w = data_a.size()

            data_a = data_a.view(-1, c, h, w)
            data_p = data_p.view(-1, c, h, w)
            data_n = data_n.view(-1, c, h, w)

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            data_a, data_p, data_n = data_a.cuda(non_blocking=True), data_p.cuda(non_blocking=True), data_n.cuda(
                non_blocking=True)

        # Compute output
        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)

        if len(data_a.size()) == 5:
            out_a = out_a.view(bs, ncrops, -1).mean(1)
            out_p = out_p.view(bs, ncrops, -1).mean(1)
            out_n = out_n.view(bs, ncrops, -1).mean(1)

        # Compute and record the loss
        loss = criterion(out_p, out_a, out_n)

        losses.update(loss.item(), data_a.size(0))

        # Log to console
        if batch_idx % log_interval == 0:
            pbar.set_description(
                'Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a),
                    len(val_loader.dataset),
                           100. * batch_idx / len(val_loader),
                    losses.avg))

        # Add mb loss to Tensorboard
        if multi_run is None:
            writer.add_scalar('val/mb_loss', loss.item(), epoch * len(val_loader) + batch_idx)
        else:
            writer.add_scalar('val/mb_loss_{}'.format(multi_run), loss.item(), epoch * len(val_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # Important, we're done evaluating loss. Turn off triplet mode.
    val_loader.dataset.train = False
    del val_loader.dataset.triplets

    # Add epoch loss to Tensorboard
    if multi_run is None:
        writer.add_scalar('val/epoch_loss', loss.item(), epoch)
    else:
        writer.add_scalar('val/epoch_loss_{}'.format(multi_run), loss.item(), epoch)

    return 0



def _evaluate_map(data_loader, model, writer, epoch, logging_label, no_cuda, log_interval, map, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    logging_label : string
        Label for logging purposes. Typically 'test' or 'valid'. Its prepended to the logging output path and messages.
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.
    map : str
        Specify value for mAP computation. Possible values are ("auto", "full" or specify K for AP@K)

    Returns
    -------
    mAP : float
        Mean average precision for evaluated on this split

    """
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    labels, outputs = [], []

    # For use with the multi-crop transform
    multi_crop = False

    # Iterate over whole evaluation set
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)
    with torch.no_grad():
        for batch_idx, (data, label) in pbar:

            # Check if data is provided in multi-crop form and process accordingly
            if len(data.size()) == 5:
                multi_crop = True
                bs, ncrops, c, h, w = data.size()
                data = data.view(-1, c, h, w)

            if not no_cuda:
                data = data.cuda()

            # Compute output
            out = model(data)

            if multi_crop:
                out = out.view(bs, ncrops, -1).mean(1)

            # Store output
            outputs.append(out.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

            # Log progress to console
            if batch_idx % log_interval == 0:
                pbar.set_description(logging_label + ' Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader)))

    # Measure accuracy (FPR95)
    num_tests = len(data_loader.dataset.file_names)
    labels = np.concatenate(labels, 0).reshape(num_tests)
    outputs = np.concatenate(outputs, 0)

    # Cosine similarity distance
    distances = pairwise_distances_chunked(outputs, metric='cosine', n_jobs=16)
    logging.debug('Computed pairwise distances')
    t = time.time()
    mAP, per_class_mAP = compute_mapk(distances, labels, k=map)
    writer.add_text('Per class mAP at epoch {}\n'.format(epoch),
                    json.dumps(per_class_mAP, indent=2, sort_keys=True))

    logging.debug('Completed evaluation of mAP in {}'.format(datetime.timedelta(seconds=int(time.time() - t))))

    logging.info('\33[91m ' + logging_label + ' set: mAP: {}\n\33[0m'.format(mAP))

    # Logging the epoch-wise accuracy
    if multi_run is None:
        writer.add_scalar(logging_label + '/mAP', mAP, epoch)
    else:
        writer.add_scalar(logging_label + '/mAP{}'.format(multi_run), mAP, epoch)

    return mAP
