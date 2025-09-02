import pandas as pd
import numpy as np
import shap
import optuna
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import openpyxl
import os
import warnings
import pickle

warnings.filterwarnings("ignore")


class ModelTrainer:
    def __init__(self, df, target_col, id_cols=[], drop_cols=[], model_type='xgboost', notes=''):
        self.df = df
        self.target_col = target_col
        self.id_cols = id_cols
        self.drop_cols = drop_cols
        self.model_type = model_type.lower()
        self.notes = notes
        self.now = datetime.now()
        self.date = self.now.strftime('%Y-%m-%d')
        self.time = self.now.strftime('%H:%M:%S')

    def prepare_data(self, test_size=0.2, random_state=42, stratifyby=None):
        # Select the stratify column only from the dataframe 
        if stratifyby is not None:
            self.stratify = True
            stratify = self.df[stratifyby]
        target = self.df[self.target_col] # Extract the target feature

        # Train test split, with stratified (if exists)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df, target, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # Collect the column to stratify by for the training data
        if stratifyby is not None:
            self.X_train_stratify = self.X_train[stratifyby]
        self.train_ids = self.X_train[self.id_cols]
        self.test_ids = self.X_test[self.id_cols]
        # Remove columns not needed for training e.g. id cols and stratify cols
        self.X_train = self.X_train.drop(columns=[self.target_col] + self.id_cols + self.drop_cols, errors='ignore')
        self.X_test = self.X_test.drop(columns=[self.target_col] + self.id_cols + self.drop_cols, errors='ignore')

        # Save the feature names from remaining features
        self.feature_names = self.X_train.columns.tolist()  # Save feature names

    def tune_hyperparams(self, n_trials=20, timeout=300):
        def objective(trial):
            params = self.suggest_params(trial)
            model = self.init_model(params)
            if self.stratify:
                stratifyby = self.X_train_stratify 
            X_train_val, X_val, y_train_val, y_val = train_test_split(self.X_train, self.y_train, stratify=stratifyby, test_size=0.2, random_state=42)
            model.fit(X_train_val, y_train_val)
            pred = model.predict(X_val)
            return mean_squared_error(y_val, pred)

        study = optuna.create_study(direction="minimize", study_name=None)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.best_params = study.best_params

    def suggest_params(self, trial):
        if self.model_type == 'xgboost':
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_int("gamma", 0, 1),
                "reg_alpha": trial.suggest_int("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_int("reg_lambda", 0, 1),
            }
        elif self.model_type == 'lightgbm':
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000), # 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15), # 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True), # 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150), # 20, 40),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 60),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 3),

                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_by_tree": trial.suggest_float("colsample_by_tree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            }
        elif self.model_type == 'catboost':
            return {
                # "iterations": trial.suggest_int("iterations", 50, 300),
                # "depth": trial.suggest_int("depth", 3, 10),
                # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),


                'depth': trial.suggest_int("depth", 3, 10),
                'learning_rate': trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                'iterations': trial.suggest_int("iterations", 300, 1500),
                'bagging_temperature': trial.suggest_float("bagging_temperature", 0.0, 1.0),
                'random_strength': trial.suggest_float("random_strength", 0.0, 1.0),
                'rsm': trial.suggest_float("rsm", 0.6, 1.0),  # Like colsample_bytree
                'border_count': trial.suggest_int("border_count", 32, 255),  # For quantisation
            }
        elif self.model_type == 'random_forest':
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10)
            }
        elif self.model_type == 'svr':
            return {
                "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
            }

    def init_model(self, params):
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(random_state=42, **params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMRegressor(random_state=42, **params)
        elif self.model_type == 'catboost':
            return cb.CatBoostRegressor(random_state=42, **params, verbose=0)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(**params)
        elif self.model_type == 'svr':
            return SVR(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train_final_model(self, use_log_target=False, custom_params={}, save_model=False):
        if use_log_target:
            self.log_y_train = np.log1p(self.y_train)
            self.log_y_test = np.log1p(self.y_test)
            y_train = self.log_y_train
            y_test = self.log_y_test
        else:
            y_train = self.y_train
            y_test = self.y_test

        params = self.best_params if hasattr(self, 'best_params') else custom_params
        self.model = self.init_model(params)
        self.model.fit(self.X_train, y_train)

        self.log_y_pred_train = self.model.predict(self.X_train)
        self.log_y_pred_test = self.model.predict(self.X_test)

        if use_log_target:
            self.y_pred_train = np.expm1(self.log_y_pred_train)
            self.y_pred_test = np.expm1(self.log_y_pred_test)
        else:
            self.y_pred_train = self.log_y_pred_train
            self.y_pred_test = self.log_y_pred_test
        if save_model:
            model_filename = f"./data/modelling/models/{datetime.now().strftime(format='%Y%m%d_%H%M')}_{self.model_type}.pkl"
            pickle.dump(self.model, open(model_filename, 'wb'))

    def evaluate(self):
        def metrics(y_true, y_pred, prefix):
            return {
                f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                f'{prefix}mae': mean_absolute_error(y_true, y_pred),
                f'{prefix}r2': r2_score(y_true, y_pred),
                f'{prefix}mape': mean_absolute_percentage_error(y_true, y_pred)
            }

        # Determine if log was used from model training
        use_log_target = (
            np.any(self.y_train <= 0)
            or np.any(self.y_test <= 0)
            or np.any(self.y_pred_train <= 0)
            or np.any(self.y_pred_test <= 0)
        ) is False and (
            np.allclose(np.log1p(self.y_train), np.log1p(np.expm1(self.y_train)))
        ) is False

        self.metrics = {}

        # Metrics in original scale
        self.metrics.update(metrics(self.y_train, self.y_pred_train, 'train_'))
        self.metrics.update(metrics(self.y_test, self.y_pred_test, ''))

        # If training was done in log scale, also compute log metrics
        if hasattr(self, "log_y_train") and hasattr(self, "log_y_pred_train"):
            self.metrics.update(metrics(self.log_y_train, self.log_y_pred_train, 'log_train_'))
            self.metrics.update(metrics(self.log_y_test, self.log_y_pred_test, 'log_'))



    def compute_feature_importances(self):
        self.importances = {}
        self.feature_names = self.X_train.columns.tolist()

        if hasattr(self.model, 'feature_importances_'):
            self.importances['builtin'] = self.model.feature_importances_

        try:
            explainer = shap.Explainer(self.model, self.X_test)
            shap_values = explainer(self.X_test)
            self.importances['shap'] = np.abs(shap_values.values).mean(axis=0)
        except:
            self.importances['shap'] = [np.nan] * len(self.feature_names)

        try:
            rfe = RFE(self.model, n_features_to_select=5)
            rfe.fit(self.X_train, self.y_train)
            self.importances['rfe'] = rfe.ranking_
        except:
            self.importances['rfe'] = [np.nan] * len(self.feature_names)

        try:
            perm = permutation_importance(self.model, self.X_test, self.y_test)
            self.importances['permutation'] = perm.importances_mean
        except:
            self.importances['permutation'] = [np.nan] * len(self.feature_names)

        try:
            if self.model_type == 'xgboost':
                gain_importance = self.model.get_booster().get_score(importance_type='gain')
                self.importances['gain'] = [gain_importance.get(f, 0) for f in self.feature_names]
            elif self.model_type == 'lightgbm':
                self.importances['gain'] = self.model.booster_.feature_importance(importance_type='gain')
            elif self.model_type == 'catboost':
                self.importances['gain'] = self.model.get_feature_importance(type='PredictionValuesChange')
        except:
            self.importances['gain'] = [np.nan] * len(self.feature_names)

    def save_results(self, model_log_path='model_results.xlsx', importance_log_path='feature_importances.xlsx'):
        row = {
            'date': self.date,
            'time': self.time,
            'model_type': self.model_type,
            'experiment_notes': self.notes,
            'conclusion': ''
        }
        row.update(self.metrics)
        row.update({f'param_{k}': v for k, v in self.model.get_params().items()})
        row['feature_names'] = ', '.join(self.feature_names)
        df_metrics = pd.DataFrame([row])

        if os.path.exists(model_log_path):
            old = pd.read_excel(model_log_path)
            df_metrics = pd.concat([old, df_metrics], ignore_index=True)
        df_metrics.to_excel(model_log_path, index=False)

        if importance_log_path != "":
            importance_df = pd.DataFrame({f: imp for f, imp in self.importances.items()}, index=self.feature_names).reset_index()
            
            importance_df.insert(1, 'date', self.date)
            importance_df.insert(2, 'time', self.time)
            importance_df.insert(3, 'model_type', self.model_type)
            importance_df.insert(4, 'notes', self.notes)

            if os.path.exists(importance_log_path):
                old = pd.read_excel(importance_log_path)
                importance_df = pd.concat([old, importance_df], ignore_index=True)
            importance_df.to_excel(importance_log_path, index=False)
