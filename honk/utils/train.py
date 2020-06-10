import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from . import model as mod

from tqdm import tqdm
from honk.utils import ConfigBuilder
from sklearn.metrics import f1_score
from KS_Strong.model import KSStrong
from torchaudio.transforms import MFCC

USE_NOISE_EVAL = True


def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


melkwargs = {'n_fft': 480,
             'hop_length': 16000 // 1000 * 10,
             'f_min': 20,
             'f_max': 4000,
             'n_mels': 40
             }

YES_WORD_INDEX = 2


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


def evaluate(config, model=None, test_loader=None):
    if not test_loader:
        train_set, eval_set, test_set = mod.SpeechDataset.splits(config)

        test_loader = data.DataLoader(
            test_set,
            batch_size=len(test_set))

    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    if USE_NOISE_EVAL:
        noiser = KSStrong(config)
        if not config["no_cuda"]:
            noiser.cuda()

    criterion = nn.CrossEntropyLoss()

    mfcc = MFCC(melkwargs=melkwargs, log_mels=True)

    model.eval()
    with torch.no_grad():
        for model_in, labels in test_loader:

            if USE_NOISE_EVAL:
                noise = noiser()
                model_in += noise

            model_in = mfcc(model_in).permute(0, 2, 1)
            print(model_in.size())

            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()

            scores = model(model_in)

            loss = criterion(scores, labels)

            accuracy, f1 = eval_metrics(scores, labels)

    print("final test accuracy: {}, test f1 {}, test loss {}".format(accuracy, f1, loss.item()))


def train(config):
    output_dir = os.path.dirname(os.path.abspath(config["output_file"]))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
    model = config["model_class"](config)
    if config["input_file"]:
        model.load(config["input_file"])

    noiser = KSStrong(config)

    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
        noiser.cuda()

    optimizer = torch.optim.SGD(noiser.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"],
                                weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()

    train_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True, drop_last=True)

    dev_loader = data.DataLoader(
        dev_set,
        batch_size=len(dev_set),
        shuffle=False)

    test_loader = data.DataLoader(
        test_set,
        batch_size=len(test_set),
        shuffle=False)

    mfcc = MFCC(melkwargs=melkwargs, log_mels=True)

    step_no = 0
    for epoch_idx in range(config["n_epochs"]):
        model.train()
        noiser.train()

        for batch_idx, (model_in, labels) in enumerate(tqdm(train_loader)):

            optimizer.zero_grad()

            noise = noiser()
            model_in_aug = model_in + noise

            model_in_aug_mfcc = mfcc(model_in_aug).permute(0, 2, 1)

            if not config["no_cuda"]:
                model_in = model_in.cuda()
                model_in_aug = model_in_aug.cuda()
                model_in_aug_mfcc = model_in_aug_mfcc.cuda()
                labels = labels.cuda()

            scores = model(model_in_aug_mfcc)
            loss = -criterion(scores, labels) + torch.nn.MSELoss()(model_in_aug, model_in)

            loss.backward()
            optimizer.step()
            step_no += 1

            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                                            nesterov=config["use_nesterov"], momentum=config["momentum"],
                                            weight_decay=config["weight_decay"])
            accuracy, f1 = eval_metrics(scores, labels)
            # TODO : tensorboard logging
            print("train step {}, accuracy {}, f1 {}, loss {}".format(step_no, accuracy, f1, loss.item()))

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            with torch.no_grad():
                model.eval()
                for model_in, labels in tqdm(dev_loader):

                    noise = noiser()
                    model_in_aug = model_in + noise

                    model_in_aug_mfcc = torch.tensor(dev_set.preprocess(model_in.cpu().numpy()))

                    if not config["no_cuda"]:
                        model_in = model_in.cuda()
                        model_in_aug = model_in_aug.cuda()
                        model_in_aug_mfcc = model_in_aug_mfcc.cuda()
                        labels = labels.cuda()

                    scores = model(model_in_aug_mfcc)
                    loss = -criterion(scores, labels) + torch.nn.MSELoss()(model_in_aug, model_in)

                    accuracy, f1 = eval_metrics(scores, labels)

                    print("eval epoch id {}, accuracy {}, f1 {}, loss {}".format(epoch_idx, accuracy, f1, loss.item()))
                    torch.save(noiser.state_dict(), "./weights/epoch_{}.pth".format(epoch_idx))


def main():
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    config, _ = parser.parse_known_args()

    global_config = dict(no_cuda=False, n_epochs=500, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10,
                         seed=0,
                         use_nesterov=False, input_file="", output_file=output_file, gpu_no=1, cache_size=32768,
                         momentum=0.9, weight_decay=0.00001)
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    parser.add_argument("--type", choices=["train", "eval"], default="train", type=str)
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)
    if config["type"] == "train":
        train(config)
    elif config["type"] == "eval":
        evaluate(config)


if __name__ == "__main__":
    main()
