# Model related imports

# Dataset related imports
from torch.utils.data import Dataset, random_split, Subset

# Data augmentation/processing imports
import pickle

# Evaluation related imports

# PyTorch related imports
import torch

# Util related imports

# Execution related imports
import os

# Datastructure related imports
import pandas as pd  # noqa
import numpy as np  # noqa
import json  # noqa

# Miscellaneous imports
from typing import Any, Optional
from pprint import pprint  # noqa
import pyfiglet


class BaseDataset(Dataset):
    """Base class for data handling"""

    def __init__(
        self: "BaseDataset",
    ):
        """Initializes the BaseDataset class.

        NOTE:
            All datasets used by this parent class must use the `self.data`
            instance variable to represent the content of your dataset

        Args:
            self (BaseDataset): Instance method for the class
        """
        self.data = []

    def __len__(self: "BaseDataset") -> int:
        """
        Returns the total number of samples in the dataset.

        Args:
            self (BaseDataset): Instance method for the class

        Returns:
            int: The size of the dataset.
        """
        return len(self.data)

    def __getitem__(
        self: "BaseDataset", idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves a data sample from the dataset

        NOTE:
            This class is not defined as children datasets may have different
            representation for the data they represent

        Args:
            self (BaseDataset): Instance method for the class
            idx (int): Index of the data sample to retrieve

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The data sample
                corresponding to the given index or the given dataset source.
                Returns a feature, label, and non-state feature, respectively.

        Raises:
            NotImplementedError1: Raised if not implemented
        """
        raise NotImplementedError

    @staticmethod
    def collater(batch: list | tuple) -> Any:
        """Classmethod that contains the processing functions to collate
        the contents of the batch

        Args:
            batch (list): Batch content from the dataloader.

        Returns:
            Any: Returns the processed batch from the dataloader

        Raises:
            NotImplementedError1: Raised if not implemented
        """
        raise NotImplementedError

    def inprocess(self: "BaseDataset", input_info: dict | list | str) -> Any:
        """Classmethod that contains the inprocessing functions to process
        batch content on the fly. The dataset instance must have the
        tokenizer defined or set before using this method.

        Args:
            self (BaseDataset): Instance method for the class
            input_info (dict | list | str): Content to inprocess

        Returns:
            Any: Returns the processed batch after inprocessing

        Raises:
            NotImplementedError: Raised if not implemented
        """
        raise NotImplementedError

    def preprocess(self: "BaseDataset", **kwargs):
        """Preprocessing step for the dataset. The dataset instance must have
        the tokenizer defined or set before using this method.

        Args:
            self (BaseDataset): Instance method for the class

        Raises:
            NotImplementedError: Raised if not implemented
        """
        raise NotImplementedError

    def split_dataset(
        self: "BaseDataset",
        train_size: float,
        valid_size: Optional[float] = None,
    ) -> list[Subset]:
        """Splits the dataset into training, validation, and optionally testing
        sets. This is done via a random selection process

        Args:
            self (BaseDataset): Instance method for the class
            train_size (float): Proportion of the dataset to include in the
                train split.
            valid_size (Optional[float]): Proportion of the dataset to include
                in the validation split. If None, the remainder of the dataset
                after the train split is used as the validation set. If
                provided, the remainder is split into validation and test sets.

        Returns:
            list[Subset]: A list containing datasets for training,
                validation, and optionally testing.
        """
        # If the val_size is not provided, use the given train_size value to
        # specify the training dataset size and the remainder as the validation
        # dataset size
        if valid_size is None:
            train_len = int(len(self) * train_size)
            val_len = len(self) - train_len
            return random_split(self, [train_len, val_len])

        # If provided, split up the portion of the data for training dictated
        # by train_size. Afterwards, split it up for the validation dataset by
        # the val_size value. The remainder is left for the testing dataset
        else:
            train_len = int(len(self) * train_size)
            val_len = int(len(self) * valid_size)
            test_len = len(self) - train_len - val_len
            if test_len < 0:
                raise ValueError(
                    "train_size and val_size sum exceed dataset size"
                )
            return random_split(self, [train_len, val_len, test_len])

    def save_dataset(self: "BaseDataset", path: str):
        """Method to save the dataset as a pickle file

        Args:
            self (BaseDataset): Instance method for the class
            path (str): Path of where to save the dataset
        """
        torch.save(self, path)

    @staticmethod
    def load_dataset(path: str) -> "BaseDataset":
        """Method to save the dataset as a pickle file. This method is static.

        Args:
            path (str): Path of where to load the dataset from
        """
        return torch.load(path)

    def save_split(
        self: "BaseDataset", split: Subset, name: str, directory: str
    ):
        """Method to save split information about a dataset

        Args:
            self (BaseDataset): Instance method for the class
            split (Subset): Split to save
            name (str): Name of the split
            directory (str): Directory of where to save the split
        """
        # Make the directory if not made already
        os.makedirs(directory, exist_ok=True)

        # Get the indices of the split
        split_indices = split.indices if isinstance(split, Subset) else split

        # Save the split
        with open(os.path.join(directory, f"split_{name}.pkl"), "wb") as f:
            pickle.dump(split_indices, f)

    def load_split(
        self: "BaseDataset", dataset: "BaseDataset", split_path: str
    ) -> Subset:
        """Method to load a split to the dataset

        Args:
            self (BaseDataset): Instance method for the class
            dataset (BaseDataset): Dataset instance to assign subsets to
            split_path (str): Path to the split

        Returns:
            Subset: Split information pulled from the file
        """
        with open(split_path, "rb") as f:
            return Subset(dataset, pickle.load(f))

    def _info(self: "BaseDataset"):
        """Debugging method for the dataset instance. Assumes that the dataset
        information is in a pandas DataFrame.

        Args:
            self (BaseDataset): Instance method for the class
        """
        print(pyfiglet.figlet_format("INFO", font="big"))
        print(
            "<!> Some information may be incorrect if called before critical "
            "points\n\n"
        )
        print("Size:", self.__len__())
        print("# Features:", self.num_features)
        print("# Labels:", self.num_labels)
        print("\n")
