import torch
from torchaudio.transforms import MFCC

from KS_Strong.model import KSStrong
from utils import eval_metrics, MELKWARGS, prepare_loaders


def test(honk_config, config, model=None):
    _, _, test_loader = prepare_loaders(honk_config)

    if not model:
        model = honk_config["model_class"](honk_config)
        model.load(honk_config["input_file"])

    if not honk_config["no_cuda"]:
        model.cuda()

    if config.use_noise:
        noiser = KSStrong(honk_config)
        if not honk_config["no_cuda"]:
            noiser.cuda()
        if config.path_noiser is not None:
            noiser.load_state_dict(torch.load(config.path_noiser))

    mfcc = MFCC(melkwargs=MELKWARGS, log_mels=True)

    model.eval()
    if config.use_noise:
        noiser.eval()
    with torch.no_grad():
        for x, labels in test_loader:

            if config.use_noise:
                noise = noiser()
                x += noise

            x_mfcc = mfcc(x).permute(0, 2, 1)

            if not honk_config["no_cuda"]:
                x_mfcc = x.cuda()
                labels = labels.cuda()

            scores = model(x_mfcc)
            accuracy, f1 = eval_metrics(scores, labels)

    print("Test accuracy: {:.2f}, test f1-score: {:.2f}".format(accuracy, f1))
