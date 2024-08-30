# Set the python path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Util imports
from utils.scripts import create_output_paths, prepare_config
from utils.dict_obj import DictObj

# Miscellaneous imports
from omegaconf import DictConfig
from tqdm import tqdm
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

    # Simplify reference to the action configuration
    action = cfg.action.pre_process_analysis

    # Save the model's output path
    OUTPUT_PATH = create_output_paths(
        os.path.join(cfg.output_path, "pre_process_evaluation")
    )

    # > TODO: ⚠️ MODIFY AS NEED BE


# Run the main function
if __name__ == "__main__":
    main()
