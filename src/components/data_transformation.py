import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import os
from haversine import haversine

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = ['neighbourhood_group', 'neighbourhood', 'room_type']
            numerical_columns = [
                'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
                'reviews_per_month', 'calculated_host_listings_count',
                'availability_365', 'distance_to_times_square', 'days_since_last_review'
            ]

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numerical_columns),
                ('cat', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting feature engineering...")

            # Distance to Times Square
            times_square = (40.7580, -73.9855)
            df['distance_to_times_square'] = df.apply(
                lambda row: haversine((row['latitude'], row['longitude']), times_square), axis=1
            )

            # Clean 'last_review'
            df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
            df['days_since_last_review'] = (
                pd.to_datetime('today') - df['last_review']
            ).dt.days.fillna(df['days_since_last_review'].median())

            # Fill missing values
            df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

            # Drop zero and extreme prices
            df = df[df['price'] > 0]
            df = df[df['price'] <= 500]
            df = df[df['minimum_nights'] <= 30]
            df = df[df['number_of_reviews'] <= 200]
            df = df[df['reviews_per_month'] <= 10]

            # Remove duplicates
            df.drop_duplicates(inplace=True)

            # Log-transform target
            df['price_log'] = np.log1p(df['price'])

            return df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train and test datasets")

            train_df = self.preprocess_feature_engineering(train_df)
            test_df = self.preprocess_feature_engineering(test_df)

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "price_log"

            input_feature_train_df = train_df.drop(columns=['price', target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=['price', target_column_name])
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation complete and object saved.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)

