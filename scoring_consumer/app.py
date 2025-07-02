import json
import yaml
import logging
import logging.config
import os

from confluent_kafka import Consumer, KafkaError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import init_db
from models import ScoringResult


def logging_create():
    with open("logging_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)


logger = logging.getLogger("db_service_fraud")


class ScoringConsumer:
    def __init__(self):
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.topic = os.getenv("KAFKA_SCORING_TOPIC", "scoring")
        self.group_id = os.getenv("KAFKA_GROUP_ID", "ml-scorer")

        db_host = os.getenv("POSTGRES_HOST", "postgres")
        db_port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "fraud_detection")
        db_user = os.getenv("POSTGRES_USER", "fraud_user")
        db_password = os.getenv("POSTGRES_PASSWORD", "fraud_password")

        self.db_url = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        self.engine = create_engine(self.db_url)
        init_db(self.engine)

        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        consumer_config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }

        self.consumer = Consumer(consumer_config)
        self.consumer.subscribe([self.topic])

        logger.info(f"Initialized scoring consumer for topic: {self.topic}")
        logger.info(f"Database URL: {self.db_url}")

    def process_message(self, message_value: str):
        """Обработка сообщения из Kafka и сохранение в БД"""
        try:
            data = json.loads(message_value)[0]
            logger.info("Get data topic %s: %s", self.topic, data)

            scoring_result = ScoringResult(
                transaction_id=data["transaction_id"],
                score=float(data["score"]),
                fraud_flag=bool(data["fraud_flag"]),
            )

            self.session.add(scoring_result)
            self.session.commit()

            logger.info("Saved scoring result in database")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.session.rollback()

    def start_consuming(self):
        """Запуск консьюмера"""
        logger.info("Starting scoring consumer...")

        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                        break

                message_value = msg.value().decode("utf-8")
                logger.debug(f"Received message: {message_value}")

                self.process_message(message_value)

        except KeyboardInterrupt:
            logger.info("Shutting down scoring consumer...")
        finally:
            self.consumer.close()
            self.session.close()


if __name__ == "__main__":
    consumer = ScoringConsumer()
    consumer.start_consuming()
