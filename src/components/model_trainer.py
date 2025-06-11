import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,              
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
            }
            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                   # "max_depth": [None, 5, 10, 15, 20],
                    #"min_samples_split": [2, 5, 10],
                },
                "Random Forest": {
                    "n_estimators": [8,16,32,64,128,256],
                 #   "max_depth": [None, 5, 10, 15, 20],
                  #  "min_samples_split": [2, 5, 10],
                },
                "Gradient Boosting": {
                    "n_estimators": [8,16,32,64,128,256],
                    "learning_rate": [.1, .01,.05,.001],
                   # "max_depth": [3, 5, 7],
                   "subsample":[0.6,0.7,0.75,0.8,0.85,0.9]
                },
                "Linear Regression": {
                    "fit_intercept": [True, False]
                },
                "KNeighbors Regressor": {
                    "n_neighbors": [5,7,9,11],
                   # "weights": ["uniform", "distance"],
                    #"algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                },
                "XGB Regressor": {
                    "n_estimators": [8,16,32,64,128,256],
                    "learning_rate": [.1, .01,.05,.001],
                    #"max_depth": [3, 5, 7],
                    #"subsample":[0.6,0.7,0.75,0.8,0.85,0.9]
                },
                "AdaBoost Regressor": {
                    "n_estimators": [8,16,32,64,128,256],
                    "learning_rate": [.1, .01,.05,.001],
                
                },
                "CatBoost Regressor": {
                    "iterations": [30,50,100],
                    "learning_rate": [.1, .01,.05,.001],
                    "depth": [6,8,10],
                    #"subsample":[0.6,0.7,0.75,0.8,0.85,0.9]
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    "No best model found with sufficient accuracy")
            logging.info(f"Best model found on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_score
            


        except Exception as e:
            raise CustomException(e, sys)

           