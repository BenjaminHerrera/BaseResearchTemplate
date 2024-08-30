# Set the python path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Model related imports
import models

# Dataset related imports
from torch.utils.data import DataLoader

# Evaluation related imports
import evaluation

# PyTorch and training related imports
import torch

# Util imports
from utils.scripts import (
    create_output_paths,
    prepare_config,
    dataset_reader,
    broadcast_string,
    build_evaluation_list,
)
from utils.evaluation import post_process
from utils.dict_obj import DictObj

# Execution related imports
from accelerate import Accelerator
import itertools

# Datastructure related imports
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
import geopandas as gpd
import pandas as pd  # noqa
import numpy as np
import json  # noqa

# Miscellaneous imports
from pprint import pprint  # noqa
from tqdm import tqdm
import pyfiglet
import hydra


@hydra.main(
    version_base=None,
    config_path="../configs/",
    config_name="fcnn_6",
)
def main(cfg: DictConfig):
    """Main run function with config wrapper

    Args:
        cfg (DictConfig): config object
    """

    # Resolve any system placeholder in the config and configurations
    cfg = prepare_config(cfg)

    # Print the configuration
    print(OmegaConf.to_yaml(cfg))

    #
    #   ? 🥾 SETUP PROCESS
    #   region
    #

    # Initialize Accelerator
    accelerator = Accelerator()

    # Simplify reference to the action configuration
    action = cfg.action.post_process_analysis

    # Save and create the model train session's output path
    OUTPUT_PATH = ""
    os.makedirs(cfg.output_path, exist_ok=True)
    if accelerator.is_main_process:
        OUTPUT_PATH = create_output_paths(
            os.path.join(cfg.output_path, "post_process_evaluation"),
            cfg.get("number", None),
        )
    accelerator.wait_for_everyone()
    OUTPUT_PATH = broadcast_string(OUTPUT_PATH, accelerator)

    # Extrapolate the datasets and get number of features, labels, etc.
    datasets = dataset_reader(cfg.dataset_path)

    # Create data loaders
    dataloaders = DictObj({})
    for i in datasets:
        dataloaders[i] = DataLoader(
            datasets[i], batch_size=cfg.action.train.single.batch_size
        )

    # Make the evaluation list for validation evaluations
    evaluation_list = build_evaluation_list(cfg.evaluation.components)

    #   endregion

    #
    #   ? 🧪 MODEL INFERENCING
    #   region
    #

    # Iterate through the models list for evaluation
    if cfg.get("specific_run", False):
        iteration = [cfg.get("specific_run")]
    else:
        iteration = os.listdir(action.target_analysis_path)
    for run in tqdm(iteration[:None], desc="Inferencing models"):
        # Get their absolute path
        path = os.path.join(action.target_analysis_path, run)

        # Extract the best model checkpoint
        for file in os.listdir(path + "/checkpoints/"):
            if "best" in file:
                checkpoint = torch.load(path + "/checkpoints/" + file)

        # Initialize a skeleton model
        model = getattr(models, f"{cfg.model.model.upper()}Model")(
            input_size=datasets.total.sample_num_features,
            output_size=datasets.total.sample_num_labels,
            **cfg.model.args,
        )

        # Load the model and set to evaluation mode
        new_model_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:] if k.startswith("module.") else k
            new_model_state_dict[name] = v
        model.load_state_dict(new_model_state_dict, strict=False)
        model.eval()

        # Run Inference
        with torch.no_grad():
            # ? Inference on the TEST dataset
            for batch in dataloaders.test:
                # Dissect, pass, and convert the data from the model output
                inputs, targets = batch
                outputs = model(inputs)

                # > TODO: ⚠️ MODIFY AS NEED BE

            # ? Inference on the VALID dataset
            for batch in dataloaders.valid:
                # Dissect, pass, and convert the data from the model output
                inputs, targets = batch
                outputs = model(inputs)

                # > TODO: ⚠️ MODIFY AS NEED BE

    #   endregion

    #
    #   ? 📊 MODEL EVALUATION
    #   > TODO: ⚠️ MODIFY AS NEED BE
    #   region
    #
    
    #   endregion

# Run the main function
if __name__ == "__main__":
    main()
