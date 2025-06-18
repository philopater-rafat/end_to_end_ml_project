import os
import sys
from dataclasses import dataclass

import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor):
        try:
            logging.info("Splitting train/test arrays")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42),
                "LightGBM": LGBMRegressor(random_state=42),
                "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
            }

            params = {
                "CatBoost": {
                    'regressor__depth': [6, 8, 10],
                    'regressor__learning_rate': [0.01, 0.05, 0.1],
                    'regressor__iterations': [50, 100]
                }
            }

            best_score = -np.inf
            best_model = None
            best_pipeline = None
            best_model_name = ""

            for name, model in models.items():
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])

                if name == "CatBoost":
                    logging.info(f"Tuning hyperparameters for {name}")
                    grid = GridSearchCV(pipeline, params[name], cv=3, scoring='r2', verbose=0, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    final_model = grid.best_estimator_
                else:
                    logging.info(f"Training {name} without hyperparameter tuning")
                    final_model = pipeline
                    final_model.fit(X_train, y_train)

                # Evaluate
                y_pred_log = final_model.predict(X_test)
                y_pred = np.expm1(y_pred_log)
                y_test_actual = np.expm1(y_test)

                r2 = r2_score(y_test, y_pred_log)
                logging.info(f"{name} R² (log): {r2}")

                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_pipeline = final_model
                    best_model_name = name

            if best_score < 0.6:
                raise CustomException("No sufficiently accurate model found")

            logging.info(f"Best model: {best_model_name} with R²: {best_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_pipeline
            )

            # Final Evaluation on original price scale
            y_final_pred = np.expm1(best_pipeline.predict(X_test))
            y_final_true = np.expm1(y_test)
            final_r2 = r2_score(y_final_true, y_final_pred)
            logging.info(f"Final R² on original scale: {final_r2}")

            return final_r2

        except Exception as e:
            raise CustomException(e, sys)
