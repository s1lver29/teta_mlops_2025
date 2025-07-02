import logging

import pandas as pd

from src.model import FraudModel

logger = logging.getLogger("fraud")


class ScoringModel:
    def __init__(self, model_path: str, **config):
        self.model = FraudModel().load_model(model_path)
        self.model_th: float = config.get("model_th", 0.5)

    def score_data(self, cleaned_df: pd.DataFrame, source_info: str):
        predict_proba_data = self.model.predict_proba(cleaned_df)[:, 1]

        submission = pd.DataFrame(
            {
                "score": predict_proba_data,
                "fraud_flag": (predict_proba_data > self.model_th).astype(int),
            }
        )

        logger.info("Prediction complete for data from %s", source_info)

        return submission
