import torch
from torchaudio.transforms import MFCC

from KS_Strong.model import KSStrong
from utils import eval_metrics, MELKWARGS, prepare_loaders


def test(config, model=None, use_noise=False):
    _, _, test_loader = prepare_loaders(config)

    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])

    if not config["no_cuda"]:
        model.cuda()

    if use_noise:
        noiser = KSStrong(config)
        if not config["no_cuda"]:
            noiser.cuda()

    mfcc = MFCC(melkwargs=MELKWARGS, log_mels=True)

    model.eval()
    if use_noise:
        noiser.eval()
    with torch.no_grad():
        for x, labels in test_loader:

            if use_noise:
                noise = noiser()
                x += noise

            x_mfcc = mfcc(x).permute(0, 2, 1)

            if not config["no_cuda"]:
                x_mfcc = x.cuda()
                labels = labels.cuda()

            scores = model(x_mfcc)
            accuracy, f1 = eval_metrics(scores, labels)

    print("Test accuracy: {}, test f1-score: {}".format(accuracy, f1))
