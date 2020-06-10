import os
import argparse
import numpy as np

import honk.utils.model as mod
from honk.utils.config import ConfigBuilder
from utils import set_seed
from test import test
from train import train


def main():
    """
    Code was adapted from the Honk project
    """
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
        test(config)


if __name__ == "__main__":
    main()
