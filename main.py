import os
import argparse
import numpy as np

import honk.utils.model as mod
from honk.utils.config import ConfigBuilder
from utils import set_seed, get_args
from test import test
from train import train


def main():
    """
    Code was adapted from the Honk project
    """
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    honk_config, _ = parser.parse_known_args()

    global_config = dict(no_cuda=False, n_epochs=500, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10,
                         seed=0,
                         use_nesterov=False, input_file="", output_file=output_file, gpu_no=1, cache_size=32768,
                         momentum=0.9, weight_decay=0.00001)
    mod_cls = mod.find_model(honk_config.model)
    builder = ConfigBuilder(
        mod.find_config(honk_config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    parser.add_argument("--type", choices=["train", "eval"], default="train", type=str)
    honk_config = builder.config_from_argparse(parser)
    honk_config["model_class"] = mod_cls
    set_seed(honk_config)

    config = get_args()

    if honk_config["type"] == "train":
        train(honk_config)
    elif honk_config["type"] == "eval":
        test(honk_config, config)


if __name__ == "__main__":
    main()
