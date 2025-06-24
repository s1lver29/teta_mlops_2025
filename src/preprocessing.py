import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import reverse_geocoder as rg
import usaddress
from geopy.distance import great_circle

logger = logging.getLogger("fraud")


@lru_cache(maxsize=128)
def calculate_distance(row_client: tuple[float], merchant_coords: tuple[float]):
    return great_circle(row_client, merchant_coords).km


@lru_cache(maxsize=256)
def get_info_coord(coord: tuple[float]):
    try:
        return rg.search(coord)
    except Exception:
        logger.exception("Error fetching info for coordinates %s", coord)
        return {}


@lru_cache(maxsize=256)
def parse_address(address: str):
    try:
        parsed = usaddress.tag(address)[0]
        return {
            "street_number": parsed.get("AddressNumber", -1),
            "street_name": parsed.get("StreetName", "unk"),
            "street_suffix": parsed.get("StreetNamePostType", "unk"),
            "apartment": int(parsed.get("OccupancyIdentifier", -1)),
            "occupancy_identifier": parsed.get("OccupancyType", "unk"),
        }
    except Exception:
        logger.exception("Error parsing address %s", address)
        return {
            "street_number": -1,
            "street_name": "unk",
            "street_suffix": "unk",
            "apartment": -1,
            "occupancy_identifier": "unk",
        }


class Preprocessor:
    user_group: ClassVar[list[str]] = ["name_1", "name_2", "one_city", "us_state", "post_code"]
    category_group: ClassVar[list[str]] = ["cat_id"]
    merch_group: ClassVar[list[str]] = ["merch"]

    def __init__(self, save_dir: str = "./.data_cache"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized Preprocessor with save_dir: %s", save_dir)

    def _save_last_records(self, df: pd.DataFrame, group_by: list, filename: str) -> None:
        logger.info("Saving last records grouped by %s to %s", group_by, filename)
        grouped = df.groupby(group_by).tail(30).reset_index()[df.columns]
        with Path.open(self.save_dir / filename, "wb") as f:
            pickle.dump(grouped, f)

    def _load_last_records(self, cache_file: str) -> pd.DataFrame:
        logger.info("Loading last records from %s", cache_file)
        with Path.open(self.save_dir / cache_file, "rb") as f:
            return pickle.load(f)  # noqa: S301

    def fit(self, df: pd.DataFrame):
        logger.info("Fitting Preprocessor and saving last records")
        self._save_last_records(df, group_by=self.user_group, filename="user_cache.pkl")
        self._save_last_records(df, group_by=self.category_group, filename="category_cache.pkl")
        self._save_last_records(df, group_by=self.merch_group, filename="merchant_cache.pkl")
        logger.info("Fit completed")

    def transform(self, df: pd.DataFrame, load_cache: bool = False) -> pd.DataFrame:
        logger.info("Transforming data")

        if load_cache:
            logger.info("Loading cached last records")
            last_records = pd.concat(
                [
                    self._load_last_records("user_cache.pkl"),
                    self._load_last_records("category_cache.pkl"),
                    self._load_last_records("merchant_cache.pkl"),
                ]
            ).drop_duplicates()

            last_records["del_rows"] = 1
            df["del_rows"] = 0

            df = pd.concat([last_records, df])
            logger.debug("Merged cached records with input data")

        df = df.copy(deep=True).sort_values("transaction_time").reset_index(drop=True)
        logger.debug("Sorted data by transaction_time")

        numeric_features = self.preprocess_numeric_features(df)
        rolling_features_by_user = self.preprocess_rolling_features_by_user(df)
        rolling_features_by_category = self.preprocess_rolling_features_by_category_id(df)
        rolling_features_by_merch = self.preprocess_rolling_features_by_merch(df)
        geolocation_features = self.preprocess_geolocation_features(df)
        address_features = self.preprocess_address_features(df)

        df = pd.concat(
            [
                df,
                numeric_features,
                rolling_features_by_user,
                rolling_features_by_category,
                rolling_features_by_merch,
                geolocation_features,
                address_features,
            ],
            axis=1,
        )
        logger.debug("Concatenated all features")

        if load_cache:
            df = df[df["del_rows"] == 0]
            df = df.drop(columns="del_rows")
            logger.debug("Filtered out cached rows")

        # Удаляем лишние столбцы
        df = df.drop(
            columns=[
                "lat",
                "lon",
                "merchant_lat",
                "merchant_lon",
                "admin2_coords",
                "merchant_admin2_coords",
                "name_coords",
                "merchant_name_coords",
                "transaction_time",
                "street",
            ],
        )
        logger.info("Dropped unnecessary columns")
        logger.info("Preprocessed data columns: %s", df.columns)

        return df

    def fit_transform(self, df: pd.DataFrame, load_cache: bool = False):
        logger.info("Running fit_transform")
        self.fit(df)
        return self.transform(df, load_cache=load_cache)

    @classmethod
    def preprocess_numeric_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing numeric features")
        return pd.DataFrame(
            {
                "population_city_log": np.log(df["population_city"] + 1),
                "amount_log": np.log(df["amount"] + 1),
            }
        )

    @classmethod
    def preprocess_rolling_features_by_user(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing rolling features by user")
        grouped = df.groupby(by=cls.user_group)
        return pd.DataFrame(
            {
                "user_mean_amount_7_transactions_prev": grouped["amount"]
                .rolling(7)
                .mean()
                .fillna(-1)
                .to_numpy(),
                "user_mean_amount_30_transactions_prev": grouped["amount"]
                .rolling(30)
                .mean()
                .fillna(-1)
                .to_numpy(),
                "prev_amount": grouped["amount"].shift(1).fillna(-1).to_numpy(),
                "prev_two_amount": grouped["amount"].shift(2).fillna(-1).to_numpy(),
                "prev_three_amount": grouped["amount"].shift(3).fillna(-1).to_numpy(),
                "prev_four_amount": grouped["amount"].shift(4).fillna(-1).to_numpy(),
                "prev_cat_id": grouped["cat_id"].shift(1).fillna("unk"),
            }
        )

    @classmethod
    def preprocess_rolling_features_by_merch(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing rolling features by merch")
        grouped = df.groupby(by=cls.merch_group)["amount"]
        return pd.DataFrame(
            {
                "mean_amount_merch_30_transactions_prev": grouped.rolling(30, min_periods=1)
                .mean()
                .to_numpy(),
                "std_amount_merch_30_transactions_prev": grouped.rolling(30, min_periods=1)
                .std()
                .to_numpy(),
            }
        ).fillna(-1)

    @classmethod
    def preprocess_rolling_features_by_category_id(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing rolling features by category ID")
        grouped = df.groupby(by=cls.category_group)["amount"]
        return pd.DataFrame(
            {
                "mean_amount_cat_30_transactions_prev": grouped.rolling(30).mean().to_numpy(),
                "std_amount_cat_30_transactions_prev": grouped.rolling(30).std().to_numpy(),
            }
        ).fillna(-1)

    @classmethod
    def preprocess_geolocation_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing geolocation features")
        df["distance_km_in_table_info"] = df.apply(
            lambda x: calculate_distance(
                (x["lat"], x["lon"]), (x["merchant_lat"], x["merchant_lon"])
            ),
            axis=1,
        )
        uniq_coord = tuple(df[["lat", "lon"]].itertuples(index=False, name=None))
        merchant_uniq_coord = tuple(
            df[["merchant_lat", "merchant_lon"]].itertuples(index=False, name=None)
        )
        dict_coords = get_info_coord(uniq_coord)
        merchant_dict_coords = get_info_coord(merchant_uniq_coord)

        coords = pd.DataFrame(dict_coords).rename(
            columns={
                "lat": "lat_coords",
                "lon": "lon_coords",
                "name": "name_coords",
                "admin1": "admin1_coords",
                "admin2": "admin2_coords",
                "cc": "cc_coords",
            }
        )[["name_coords", "admin2_coords"]]

        merchant_coords = pd.DataFrame(merchant_dict_coords).rename(
            columns={
                "lat": "merchant_lat_coords",
                "lon": "merchant_lon_coords",
                "name": "merchant_name_coords",
                "admin1": "merchant_admin1_coords",
                "admin2": "merchant_admin2_coords",
                "cc": "merchant_cc_coords",
            }
        )[["merchant_name_coords", "merchant_admin2_coords"]]

        return pd.concat([coords, merchant_coords], axis=1)

    @classmethod
    def preprocess_address_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessing address features")
        return df["street"].apply(lambda x: pd.Series(parse_address(x)))
