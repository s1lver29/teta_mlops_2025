import logging
import logging.config
import time
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from src.model import FraudModel
from src.preprocessing import Preprocessor
from src.score import score_data
from src.train import TraningModel


def configure_logging(cfg_logging: DictConfig):
    logging.config.dictConfig(OmegaConf.to_container(cfg_logging, resolve=True))  # type: ignore


logger = logging.getLogger("fraud")


class InputFileHandler(FileSystemEventHandler):
    def __init__(self, model_path: str, output_dir: str, train_data_path: str):
        self.model_path = model_path
        self.output_dir = output_dir
        self.train_data_path = train_data_path

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".csv"):
            return

        input_path = str(event.src_path)

        logger.info("Detected file for scoring: %s", input_path)

        # try:
        train_data = (
            pd.read_csv(self.train_data_path).drop(columns="target").sample(1000).reset_index()
        )
        data = pd.read_csv(input_path).sample(1000).reset_index()

        train_data["test_columns"] = 0
        data["test_columns"] = 1

        preprocessor = Preprocessor()
        preprocessed_data = preprocessor.transform(pd.concat([train_data[data.columns], data]))

        preprocessed_data = (
            preprocessed_data[preprocessed_data["test_columns"] == 1]
            .sort_values(by="index")
            .drop(columns=["test_columns", "index"])
        )

        logging.info("Scoring data")
        score_data(
            preprocessed_data,
            self.model_path,
            input_path,
            self.output_dir,
            additional_outputs=True,
        )


def train_model(cfg_model_train: DictConfig):
    logger.info("Train model")
    try:
        preprocessor = Preprocessor()

        model = FraudModel(**cfg_model_train.model_params)
        logger.info("Init model with params %s", cfg_model_train.model_params)
        training_model = TraningModel(preprocessor, model)

        if not Path(cfg_model_train.train_data_path).exists():
            raise FileNotFoundError(  # noqa: TRY003, TRY301
                f"Training data not found at path: {cfg_model_train.train_data_path}"
            )

        train_data = pd.read_csv(cfg_model_train.train_data_path)

        training_model.train(
            train_data, cfg_model_train.target_column, cfg_model_train.save_model_path
        )
        logger.info("Model successfully trained and saved to %s", cfg_model_train.save_model_path)
    except Exception:
        logger.exception("Error during model training")
        raise


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    configure_logging(cfg.logging)
    try:
        if cfg.train:
            train_model(cfg)
        elif not Path(cfg.model_path).exists():
            raise FileNotFoundError(  # noqa: TRY003, TRY301
                f"Model not found at {cfg.model_path}. "
                "Please provide a valid path or train the model."
            )

        input_dir = cfg.input_dir
        output_dir = cfg.output_dir

        if not Path(input_dir).exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")  # noqa: TRY003, TRY301

        event_handler = InputFileHandler(
            model_path=cfg.model_path,
            output_dir=output_dir,
            train_data_path=cfg.train_data_path,
        )
        observer = Observer()
        observer.schedule(event_handler, path=input_dir, recursive=False)

        logger.info("Waiting for files in the input directory...")
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    except Exception:
        logger.exception("Error in the main function")
        raise


if __name__ == "__main__":
    main()
