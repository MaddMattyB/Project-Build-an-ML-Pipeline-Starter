import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
import argparse
import yaml

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


@hydra.main(version_base=None, config_path="home/mbazzle2012/Project-Build-an-ML-Pipeline-Starter/config", config_name="config")
def main(cfg):
    # Access parameters from the config
    min_price = cfg.data.min_price
    max_price = cfg.data.max_price

    # Call the basic cleaning step
    clean_data = basic_cleaning(raw_data, min_price=min_price, max_price=max_price)

    # Continue with the rest of your pipeline
    ...

if __name__ == "__main__":
    main()
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run the ML pipeline.')
    parser.add_argument('--input_artifact', type=str, required=True, help='Path to the input artifact')
    parser.add_argument('--output_artifact', type=str, required=True, help='Name for the output artifact')
    parser.add_argument('--output_type', type=str, required=True, help='Type of the output artifact')
    parser.add_argument('--output_description', type=str, required=True, help='Description of the output artifact')
    parser.add_argument('--min_price', type=float, required=True, help='Minimum price for filtering')
    parser.add_argument('--max_price', type=float, required=True, help='Maximum price for filtering')

    args = parser.parse_args()

    # Load configuration
    config = load_config('config.yaml')

    # Now you can access the parameters from args
    min_price = args.min_price
    max_price = args.max_price

    

 
    ...

if __name__ == "__main__":
    main()

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
             _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "data_split" in active_steps:
           _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                'main',
                parameters = {
                    "input": "clean_sample.csv:latest",
                    "test_size": config['modeling']['test_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by']
                }
            )
        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "rf_config": rf_config,
                    "max_tfidf_features": config['modeling']['max_tfidf_features'],
                    "output_artifact": "random_forest_export"
                },
            )


        if "test_regression_model" in active_steps:

            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                'main',
                parameters = {
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest"
                }
            )


if __name__ == "__main__":
    go()
