import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from torchaudio.transforms import MFCC
from tensorboardX import SummaryWriter

from utils import eval_metrics, MELKWARGS, prepare_loaders
from KS_Strong.model import KSStrong


def train(honk_config):
    print("Training: Begin")

    writer = SummaryWriter()

    train_loader, eval_loader, _ = prepare_loaders(honk_config)

    model = honk_config["model_class"](honk_config)
    if honk_config["input_file"]:
        model.load(honk_config["input_file"])

    noiser = KSStrong(honk_config)

    if not honk_config["no_cuda"]:
        model.cuda()
        noiser.cuda()

    optimizer = torch.optim.Adam(noiser.parameters())
    criterion = nn.CrossEntropyLoss()

    mfcc = MFCC(melkwargs=MELKWARGS, log_mels=True)

    for epoch_idx in range(honk_config["n_epochs"]):

        print("Epoch {}/{}".format(epoch_idx, honk_config["n_epochs"]))

        losses = []
        accuracies = []

        model.train()
        noiser.train()

        for batch_idx, (x, labels) in enumerate(tqdm(train_loader)):

            optimizer.zero_grad()

            noise = noiser()
            x_aug = x + noise

            x_aug_mfcc = mfcc(x_aug).permute(0, 2, 1)

            if not honk_config["no_cuda"]:
                x = x.cuda()
                x_aug = x_aug.cuda()
                x_aug_mfcc = x_aug_mfcc.cuda()
                labels = labels.cuda()

            scores = model(x_aug_mfcc)

            loss = -criterion(scores, labels) + torch.nn.MSELoss()(x_aug, x)
            loss.backward()
            optimizer.step()

            accuracy, _ = eval_metrics(scores, labels)
            losses.append(loss.item())
            accuracies.append(accuracy)

        print("Train epoch {}, loss {:.2f}, accuracy {:.2f}".format(epoch_idx, -np.mean(losses), np.mean(accuracies)))
        writer.add_scalar("Train/loss", -np.mean(losses), epoch_idx)
        writer.add_scalar("Train/accuracy", np.mean(accuracies), epoch_idx)

        model.eval()
        with torch.no_grad():
            for x, labels in eval_loader:

                noise = noiser()
                x_aug = x + noise

                x_aug_mfcc = mfcc(x_aug).permute(0, 2, 1)

                if not honk_config["no_cuda"]:
                    x = x.cuda()
                    x_aug = x_aug.cuda()
                    x_aug_mfcc = x_aug_mfcc.cuda()
                    labels = labels.cuda()

                scores = model(x_aug_mfcc)
                loss = -criterion(scores, labels) + torch.nn.MSELoss()(x_aug, x)
                accuracy, f1 = eval_metrics(scores, labels)

                print("Val.  epoch {}, loss {:.2f}, accuracy {:.2f}, f1-score {:.2f}".format(epoch_idx,
                                                                                             -loss.item(),
                                                                                             accuracy,
                                                                                             f1))
                writer.add_scalar("Eval/loss", -loss.item(), epoch_idx)
                writer.add_scalar("Eval/accuracy", accuracy, epoch_idx)
                writer.add_scalar("Eval/f1", f1, epoch_idx)

        torch.save(noiser.state_dict(), "./weights/epoch_{}.pth".format(epoch_idx))

    print("Training: Done")
