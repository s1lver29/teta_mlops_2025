import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import Pool

from src.model import FraudModel

logger = logging.getLogger("fraud")


def save_feature_importances(
    cleaned_df: pd.DataFrame, model: FraudModel, feature_importance_json: str | Path
):
    """
    Save the top-5 feature importances of the model to a JSON file.

    Args:
        cleaned_df (pd.DataFrame): Preprocessed data for scoring.
        model_path (str): Path to the trained model file.
        feature_importance_json (str): Path to save the top-5 feature importances in JSON format.
    """
    cat_features = cleaned_df.iloc[:, model.model.get_cat_feature_indices()].columns.tolist()
    text_features = cleaned_df.iloc[:, model.model.get_text_feature_indices()].columns.tolist()
    pool = Pool(data=cleaned_df, cat_features=cat_features, text_features=text_features)

    feature_importances = model.model.get_feature_importance(pool, prettified=True)
    logger.info(feature_importances.head(5))  # type: ignore

    with Path.open(feature_importance_json, "w") as f:  # type: ignore
        json.dump(
            feature_importances.head(5)  # type: ignore
            .set_index("Feature Id")["Importances"]
            .to_dict(),
            f,
            indent=4,
        )
    logger.info("Feature importances saved to %s", feature_importance_json)


def plot_prediction_density(predictions: np.ndarray, density_plot_path: str | Path):
    """
    Создание и сохранение графика плотности распределения предсказанных скоров.

    Args:
        predictions (pd.DataFrame): Предсказанные вероятности модели.
        density_plot_path (str | Path): Путь для сохранения графика плотности.
    """
    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        predictions[:, 0],
        fill=True,
        color="green",
        alpha=0.5,
        label="Не фрод",  # noqa: RUF001
        log_scale=True,
    )

    sns.kdeplot(
        predictions[:, 1],
        fill=True,
        color="red",
        alpha=0.5,
        label="Фрод",
        log_scale=True,
    )

    plt.title("График плотности распределения предсказанных скоров", fontsize=14)
    plt.xlabel("Скор предсказания", fontsize=12)
    plt.ylabel("Плотность", fontsize=12)
    plt.legend(loc="upper right", fontsize=10)

    plt.savefig(density_plot_path)
    logger.info("Density plot saved to %s", density_plot_path)
    plt.close()


def score_data(
    cleaned_df: pd.DataFrame,
    path_to_file: str,
    input_file_name: str,
    output_path: str,
    additional_outputs: bool = False,
):
    model_th: float = 0.5

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_file_name).stem

    out_path = Path.cwd() / output_path
    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Path output: %s", out_path)

    model = FraudModel()
    model.load_model(path_to_file)

    predict_proba = model.predict_proba(cleaned_df)

    predictions_file = out_path / f"{base_name}_predictions_{timestamp}.csv"  # type: ignore
    predictions = pd.DataFrame(
        {
            "index": cleaned_df.index,
            "prediction": (predict_proba[:, 1] > model_th).astype(int),
        }
    )
    predictions.to_csv(predictions_file, index=False)
    logger.info("Scoring results saved to %s", predictions_file)

    if additional_outputs:
        save_feature_importances(
            cleaned_df,
            model,
            out_path / f"{base_name}_feature_importances_{timestamp}.json",
        )
        plot_prediction_density(
            predict_proba, out_path / f"{base_name}_density_plot_{timestamp}.png"
        )
