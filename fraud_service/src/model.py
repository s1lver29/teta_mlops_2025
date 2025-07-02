import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

logger = logging.getLogger("fraud")


class FraudModel:
    def __init__(self, **config):
        self.model = CatBoostClassifier(**config)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame | None = None,
        y_test: pd.DataFrame | None = None,
        **config_fit,
    ):
        if X_test is not None and y_test is not None:
            self.model.fit(X_train, y_train, eval_set=(X_test, y_test), **config_fit)
            logging.info(
                "F1 score validation: %s",
                np.round(f1_score(y_test, self.model.predict(X_test), average="macro"), 5),
            )
        else:
            self.model.fit(X_train, y_train, **config_fit)

    def predict(self, X: pd.DataFrame, **config_predict):
        return self.model.predict(X, **config_predict)

    def predict_proba(self, X: pd.DataFrame, **config_predict):
        return self.model.predict_proba(X, **config_predict)

    def set_params(self, **params):
        if self.model.is_fitted():
            self.model = CatBoostClassifier(**self.model.get_params())

        self.model.set_params(**params)

    def save_model(self, path_save_model: str):
        self.model.save_model(path_save_model)

    def load_model(self, path_load_model: str):
        return self.model.load_model(path_load_model)
