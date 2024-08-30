# Model related imports

# Dataset related imports
from dataset import BaseDataset

# Data augmentation/processing imports

# Evaluation related imports

# PyTorch related imports
import torch

# Util related imports

# Execution related imports

# Datastructure related imports

# Miscellaneous imports
from typing import Any


class ExampleDataset(BaseDataset):
    """Example dataset for demonstration purposes"""

    def __init__(self: "ExampleDataset", arg1: Any):
        """Constructor for the dataset

        Args:
            self (ExampleDataset): Instance method for the class
            arg1 (Any): Example argument
        """

        # Make a random representative dataset
        self.data = [[2, 4], [5, 7], [2, 3]]

    def __getitem__(self: "ExampleDataset", idx: int) -> list[int, int]:
        """Retrieves a data sample from the dataset

        Args:
            self (BaseDataset): Instance method for the class
            idx (int): Index of the data sample to retrieve

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The data sample
                corresponding to the given index or the given dataset source.
                Returns a feature, label, and non-state feature, respectively.
        """
        return torch.tensor([self.data[idx][1]]), torch.tensor(
            [self.data[idx][1]]
        )
