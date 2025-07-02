import json
import logging
import logging.config
import os
from pathlib import Path

import hydra
import pandas as pd
from confluent_kafka import Consumer, KafkaError, Producer
from omegaconf import DictConfig, OmegaConf

from src.model import FraudModel
from src.preprocessing import Preprocessor
from src.score import ScoringModel
from src.train import TraningModel


def configure_logging(cfg_logging: DictConfig):
    logging.config.dictConfig(OmegaConf.to_container(cfg_logging, resolve=True))  # type: ignore


logger = logging.getLogger("fraud")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scoring")


class KafkaDataHandler:
    def __init__(self, model_path: str, train_data_path: str):
        self.fraud_model = ScoringModel(model_path)
        self.preprocessor = Preprocessor()
        self.train_data = pd.read_csv(train_data_path)
        self.train_data["transaction_time"] = pd.to_datetime(self.train_data["transaction_time"])
        self.train_data = (
            self.train_data.sort_values("transaction_time")
            .drop(columns="target")
            .reset_index(drop=True)
        )

        # Настройка Kafka Consumer
        consumer_config = {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "ml-scorer",
            "auto.offset.reset": "earliest",
        }

        # Настройка Kafka Producer
        producer_config = {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        }

        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([TRANSACTIONS_TOPIC])
        self.producer = Producer(producer_config)

        self.output_topic = SCORING_TOPIC

    def delivery_report(self, err, msg):
        """Callback для отчета о доставке сообщения"""  # noqa: RUF002
        if err is not None:
            logger.error("Message delivery failed: %s", err)
        else:
            logger.debug("Message delivered to %s [%s]", msg.topic(), msg.partition())

    def send_results_to_kafka(self, results: pd.DataFrame):
        """Отправка результатов в Kafka"""
        try:
            result_message = results.to_json(orient="records")

            self.producer.produce(
                topic=self.output_topic,
                value=result_message,
            )

            logger.info("Sent result to Kafka: %s", result_message)

            self.producer.flush()

        except Exception as e:
            logger.error("Error sending results to Kafka: %s", e)
            logger.exception("Full error traceback")

    def process_message(self, message_value: str):
        """Обработка сообщения из Kafka"""
        try:
            # Парсим JSON сообщение
            message_data = json.loads(message_value)

            transaction_id = message_data["transaction_id"]
            input_df = pd.DataFrame([message_data["data"]])
            input_df["transaction_time"] = pd.to_datetime(input_df["transaction_time"])

            logger.info("Processing data from Kafka, shape: %s", input_df.shape)

            # Загружаем тренировочные данные для preprocessing
            train_data = self.train_data[
                self.train_data["transaction_time"] < input_df["transaction_time"][0]
            ]

            train_data["test_columns"] = 0
            input_df["test_columns"] = 1

            preprocessed_data = self.preprocessor.transform(
                pd.concat([train_data[input_df.columns], input_df])
            )

            # preprocessed_data = preprocessed_data[preprocessed_data["test_columns"] == 1].drop(
            #     columns=["test_columns"]
            # )
            logger.info(
                "Preprocessed data: shape: %s\n data: %s",
                preprocessed_data.shape,
                preprocessed_data.to_numpy(),
            )

            logger.info("Scoring data from Kafka")

            results = self.fraud_model.score_data(preprocessed_data, "scoring")
            results["transaction_id"] = transaction_id

            self.send_results_to_kafka(results)

            logger.info("Successfully processed Kafka message")

        except Exception as e:
            logger.error("Error processing Kafka message: %s", str(e))
            logger.exception("Full error traceback")

    def start_consuming(self):
        """Запуск консьюмера Kafka"""
        logger.info("Starting Kafka consumer for topic: %s", TRANSACTIONS_TOPIC)
        logger.info("Will send results to topic: %s", self.output_topic)

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info("Reached end of partition")
                        continue
                    logger.error("Kafka error: %s", msg.error())
                    break

                # Обрабатываем сообщение
                message_value = msg.value().decode("utf-8")
                logger.info(
                    "Received Kafka message: %s",
                    message_value[:100] + "..." if len(message_value) > 100 else message_value,
                )

                self.process_message(message_value)

        except KeyboardInterrupt:
            logger.info("Shutting down Kafka consumer...")
        finally:
            self.producer.flush()


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

        kafka_handler = KafkaDataHandler(
            model_path=cfg.model_path,
            train_data_path=cfg.train_data_path,
        )

        logger.info("Starting Kafka data processing...")
        kafka_handler.start_consuming()

    except Exception:
        logger.exception("Error in the main function")
        raise


if __name__ == "__main__":
    main()
