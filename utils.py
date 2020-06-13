import argparse
import torch
import random
import numpy as np
import torch.utils.data as data

import honk.utils.model as mod
from sklearn.metrics import f1_score


MELKWARGS = {'n_fft': 480,
             'hop_length': 16000 // 1000 * 10,
             'f_min': 20,
             'f_max': 4000,
             'n_mels': 40
             }

YES_WORD_INDEX = 2


def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_noise", default=False, type=bool)
    parser.add_argument("--path_noiser", default=None, type=str)
    return parser.parse_known_args()[0]


def eval_metrics(scores, labels):
    batch_size = labels.size(0)

    preds = torch.max(scores, 1)[1].view(batch_size).data
    preds[preds != YES_WORD_INDEX] = 0.0
    preds[preds == YES_WORD_INDEX] = 1.0

    gt = labels.data
    gt[gt != YES_WORD_INDEX] = 0.0
    gt[gt == YES_WORD_INDEX] = 1.0

    f1 = f1_score(gt, preds)
    accuracy = (preds == gt).float().sum() / batch_size

    return accuracy.item(), f1


def prepare_loaders(config):
    train_set, eval_set, test_set = mod.SpeechDataset.splits(config)

    train_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True)

    eval_loader = data.DataLoader(
        eval_set,
        batch_size=len(eval_set),
        shuffle=False)

    test_loader = data.DataLoader(test_set,
                                  batch_size=len(test_set))

    return train_loader, eval_loader, test_loader
