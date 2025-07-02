import pandas as pd
from sklearn.model_selection import train_test_split


class TraningModel:
    def __init__(self, preprocessing_data, model):
        self.model = model
        self.preprocessing = preprocessing_data

    def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        path_save_model: str | None = None,
        retrain_best_iterations: bool = True,
    ):
        df = df.copy(deep=True).sort_values("transaction_time").reset_index(drop=True)

        df = self.preprocessing.transform(df, load_cache=False)

        X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
            df.drop(columns=target_column),
            df[target_column],
            test_size=0.2,
            stratify=df[target_column],
            random_state=42,
            shuffle=True,
        )

        self.text_features = ["jobs"]

        self.cat_features = [
            column
            for column in X_train.select_dtypes(include=["object", "category"]).columns.tolist()
            if column not in self.text_features
        ]
        self.model.fit(
            X_train,
            y_train,
            X_test,
            y_test,
            text_features=self.text_features,
            cat_features=self.cat_features,
        )

        if retrain_best_iterations:
            X_train, y_train = df.drop(columns=target_column), df[target_column]  # noqa: N806

            self.model.set_params(iterations=self.model.model.best_iteration_)
            self.model.fit(
                X_train,
                y_train,
                text_features=self.text_features,
                cat_features=self.cat_features,
            )

        if path_save_model:
            self.model.save_model(path_save_model)
