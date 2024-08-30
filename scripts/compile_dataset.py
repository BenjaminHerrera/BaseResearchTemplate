# Set the python path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# Model related imports

# Dataset related imports
import dataset

# Data augmentation/processing imports

# Evaluation related imports

# PyTorch related imports

# Util related imports
from utils.scripts import create_output_paths, prepare_config

# Execution related imports
import hydra
import json
import copy

# Datastructure related imports
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Miscellaneous imports
import random


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

    #
    #   ? 🥾 SETUP PROCESS
    #   region
    #

    # Resolve any system placeholder in the config and configurations
    cfg = prepare_config(cfg)

    # Save the model's output path
    OUTPUT_PATH = create_output_paths(
        os.path.join(cfg.output_path, "dataset_compilation")
    )

    #   endregion

    #
    #   ? 💽 DATASET COMPILATION
    #   > TODO: ⚠️ MODIFY AS NEED BE
    #   region
    #

    # Create a dataset out of the components
    data: dataset.BaseDataset = getattr(dataset, cfg.dataset.dataset)(
        **cfg.dataset.args
    )

    # Save the entire dataset
    data.data.to_csv(f"{OUTPUT_PATH}/scaled_dataset.csv", index=False)
    data.original_data.to_csv(
        f"{OUTPUT_PATH}/original_dataset.csv", index=False
    )
    data.save_dataset(os.path.join(OUTPUT_PATH, "dataset.pth"))

    # Print the information about the dataset curation
    data._info()

    # > Example code for making splits
    """
    # Split the dataset
    train, valid, test = data.split_dataset(**cfg.dataset.split_args)

    # Save the splits in both .pkl format and the formatted dataset in .pth
    data.save_split(train, "train", OUTPUT_PATH)
    data.save_split(valid, "valid", OUTPUT_PATH)
    data.save_split(test, "test", OUTPUT_PATH)
    train_data = copy.deepcopy(data)
    valid_data = copy.deepcopy(data)
    test_data = copy.deepcopy(data)
    train_data.data = train.dataset.data.iloc[train.indices]
    valid_data.data = valid.dataset.data.iloc[valid.indices]
    test_data.data = test.dataset.data.iloc[test.indices]
    train_data.save_dataset(os.path.join(OUTPUT_PATH, "dataset.train.pth"))
    valid_data.save_dataset(os.path.join(OUTPUT_PATH, "dataset.valid.pth"))
    test_data.save_dataset(os.path.join(OUTPUT_PATH, "dataset.test.pth"))

    # Save the splits
    train.dataset.data.iloc[train.indices].to_csv(
        f"{OUTPUT_PATH}/scaled_dataset.train.csv", index=False
    )
    valid.dataset.data.iloc[valid.indices].to_csv(
        f"{OUTPUT_PATH}/scaled_dataset.valid.csv", index=False
    )
    test.dataset.data.iloc[test.indices].to_csv(
        f"{OUTPUT_PATH}/scaled_dataset.test.csv", index=False
    )
    """

    #   endregion


# Run the main function
if __name__ == "__main__":
    main()
