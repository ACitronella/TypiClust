"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import numpy as np
import errno
from byol_pytorch import BYOL
import matplotlib.pyplot as plt

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        if isinstance(model, BYOL):
            output, pre_last = model(images, return_embedding=True, return_projection=True)
        else:
            output, pre_last = model(images, return_pre_last=True)
        memory_bank.update(output, pre_last)
        if i % 25 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))

@torch.no_grad()
def fill_memory_bank_pe(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        frame_idx = batch.get("frame_idx", None)
        if frame_idx is None:
            output, pre_last = model(images, return_pre_last=True)
        else:
            output, pre_last = model(images, frame_idx, return_pre_last=True)
        memory_bank.update(output, pre_last)
        if i % 25 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)
    
    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)
    
    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' %(100*z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics(metrics: dict, title:str="loss", show=True, save_path=None):
    # `losses` and `ious` are dict that has key as label of the plot, and values as list of loss/iou values
    n_epoch = np.amin([len(l) for l in metrics.values()])
    epochs = range(1, n_epoch+1)
    plt.title(title)
    for legend, metric_values in metrics.items():
        plt.plot(epochs, metric_values, "-o", label=legend, alpha=0.4)
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
        plt.close("all")