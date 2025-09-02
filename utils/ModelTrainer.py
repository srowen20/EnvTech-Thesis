import pandas as pd
import numpy as np
import shap
import optuna
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import openpyxl  # ensure Excel writing works via pandas
import os
import warnings
import pickle

warnings.filterwarnings("ignore")


class ModelTrainer:
    '''
    A reusable class for training, tuning, evaluating, and logging ML regressors
    on tabular data (e.g., soil property prediction).

    Parameters:
        df (pd.DataFrame): Input dataframe containing features + target.
        target_col (str): Column name of the target variable.
        id_cols (list, optional): Columns to retain for reference but exclude from training.
        drop_cols (list, optional): Extra columns to drop from features.
        model_type (str): One of {'xgboost','lightgbm','catboost','random_forest','svr'}.
        notes (str): Free-text notes saved alongside logs and importances.

    Returns:
        None
    '''

    def __init__(self, df, target_col, id_cols=None, drop_cols=None, model_type='xgboost', notes=''):
        '''
        Initialises the ModelTrainer and stores metadata.

        Parameters:
            df (pd.DataFrame): Input dataframe with target and features.
            target_col (str): Name of the target column to predict.
            id_cols (list, optional): Columns to keep for audit but drop from features.
            drop_cols (list, optional): Additional columns to exclude from features.
            model_type (str): Model family to use.
            notes (str): Notes to record in logs.

        Returns:
            None
        '''
        self.df = df
        self.target_col = target_col
        self.id_cols = id_cols or []
        self.drop_cols = drop_cols or []
        self.model_type = model_type.lower()
        self.notes = notes

        # timestamps for logs
        self.now = datetime.now()
        self.date = self.now.strftime('%Y-%m-%d')
        self.time = self.now.strftime('%H:%M:%S')

        # stratification flag (set in prepare_data if used)
        self.stratify = False

    def prepare_data(self, test_size=0.2, random_state=42, stratifyby=None):
        '''
        Splits data into train/test sets and builds X/y matrices, dropping
        target/ID/extra columns from features.

        Parameters:
            test_size (float): Fraction of rows for the test split.
            random_state (int): Seed for reproducibility.
            stratifyby (str, optional): Column to stratify by (e.g., land cover).

        Returns:
            None
        '''
        stratify = None
        if stratifyby is not None:
            self.stratify = True
            stratify = self.df[stratifyby]

        target = self.df[self.target_col]

        # outer split (optionally stratified)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df, target, test_size=test_size, random_state=random_state, stratify=stratify
        )

        # keep reference IDs for audit
        self.train_ids = self.X_train[self.id_cols] if self.id_cols else pd.DataFrame(index=self.X_train.index)
        self.test_ids = self.X_test[self.id_cols] if self.id_cols else pd.DataFrame(index=self.X_test.index)

        if stratifyby is not None:
            self.X_train_stratify = self.X_train[stratifyby]

        # drop non-feature columns from X
        to_drop = [self.target_col] + self.id_cols + self.drop_cols
        self.X_train = self.X_train.drop(columns=to_drop, errors='ignore')
        self.X_test = self.X_test.drop(columns=to_drop, errors='ignore')

        self.feature_names = self.X_train.columns.tolist()

    def tune_hyperparams(self, n_trials=20, timeout=300):
        '''
        Runs Optuna hyperparameter optimisation to minimise validation MSE
        on an inner train/validation split.

        Parameters:
            n_trials (int): Number of Optuna trials to run.
            timeout (int): Soft time limit for the optimisation (seconds).

        Returns:
            None
        '''
        def objective(trial):
            params = self.suggest_params(trial)
            model = self.init_model(params)

            inner_strat = self.X_train_stratify if getattr(self, "stratify", False) else None
            X_train_val, X_val, y_train_val, y_val = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=inner_strat
            )

            model.fit(X_train_val, y_train_val)
            pred = model.predict(X_val)
            return mean_squared_error(y_val, pred)

        study = optuna.create_study(direction="minimize", study_name=None)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.best_params = study.best_params

    def suggest_params(self, trial):
        '''
        Suggests model-specific hyperparameter search spaces for Optuna.

        Parameters:
            trial (optuna.trial.Trial): The current Optuna trial.

        Returns:
            dict: A dictionary of hyperparameters for the current model family.
        '''
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
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 60),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 3),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_by_tree": trial.suggest_float("colsample_by_tree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
            }
        elif self.model_type == 'catboost':
            return {
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                "iterations": trial.suggest_int("iterations", 300, 1500),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
                "rsm": trial.suggest_float("rsm", 0.6, 1.0),  # like colsample_bytree
                "border_count": trial.suggest_int("border_count", 32, 255),
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
        '''
        Instantiates a model object for the chosen family with provided params.

        Parameters:
            params (dict): Hyperparameters to initialise the model.

        Returns:
            object: A scikit-learn-compatible regressor instance.
        '''
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

    def train_final_model(self, use_log_target=False, custom_params=None, save_model=False):
        '''
        Trains the final model on the full training set (optionally with a log
        transform of the target), using tuned params if available.

        Parameters:
            use_log_target (bool): If True, fit in log1p-space and invert for outputs.
            custom_params (dict, optional): Params to use if no tuning was run.
            save_model (bool): If True, serialise the trained model to disk.

        Returns:
            None
        '''
        custom_params = custom_params or {}

        # optional log transform of y
        if use_log_target:
            self.log_y_train = np.log1p(self.y_train)
            self.log_y_test = np.log1p(self.y_test)
            y_train = self.log_y_train
            y_test = self.log_y_test
        else:
            y_train = self.y_train
            y_test = self.y_test

        params = getattr(self, 'best_params', custom_params)
        self.model = self.init_model(params)
        self.model.fit(self.X_train, y_train)

        # predictions (stored both as raw "log space" and inverted if needed)
        self.log_y_pred_train = self.model.predict(self.X_train)
        self.log_y_pred_test = self.model.predict(self.X_test)

        if use_log_target:
            self.y_pred_train = np.expm1(self.log_y_pred_train)
            self.y_pred_test = np.expm1(self.log_y_pred_test)
        else:
            self.y_pred_train = self.log_y_pred_train
            self.y_pred_test = self.log_y_pred_test

        if save_model:
            model_filename = f"./data/modelling/models/{datetime.now().strftime('%Y%m%d_%H%M')}_{self.model_type}.pkl"
            pickle.dump(self.model, open(model_filename, 'wb'))

    def evaluate(self):
        '''
        Computes metrics on train and test sets (RMSE, MAE, R², MAPE). If the
        model was trained with a log target, also logs metrics in log space.

        Parameters:
            None

        Returns:
            None
        '''
        def metrics(y_true, y_pred, prefix):
            return {
                f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                f'{prefix}mae': mean_absolute_error(y_true, y_pred),
                f'{prefix}r2': r2_score(y_true, y_pred),
                f'{prefix}mape': mean_absolute_percentage_error(y_true, y_pred)
            }

        self.metrics = {}
        # original scale metrics
        self.metrics.update(metrics(self.y_train, self.y_pred_train, 'train_'))
        self.metrics.update(metrics(self.y_test, self.y_pred_test, ''))

        # log metrics (only if present from log training)
        if hasattr(self, "log_y_train") and hasattr(self, "log_y_pred_train"):
            self.metrics.update(metrics(self.log_y_train, self.log_y_pred_train, 'log_train_'))
            self.metrics.update(metrics(self.log_y_test, self.log_y_pred_test, 'log_'))

    def compute_feature_importances(self):
        '''
        Computes multiple importance measures:
        - built-in (tree-based models)
        - SHAP (mean absolute values on test set)
        - RFE (feature ranking)
        - permutation importance
        - gain (model-specific, if available)

        Parameters:
            None

        Returns:
            None
        '''
        self.importances = {}
        self.feature_names = self.X_train.columns.tolist()

        # built-in (if available)
        if hasattr(self.model, 'feature_importances_'):
            self.importances['builtin'] = self.model.feature_importances_

        # SHAP
        try:
            explainer = shap.Explainer(self.model, self.X_test)
            shap_values = explainer(self.X_test)
            self.importances['shap'] = np.abs(shap_values.values).mean(axis=0)
        except Exception:
            self.importances['shap'] = [np.nan] * len(self.feature_names)

        # RFE
        try:
            rfe = RFE(self.model, n_features_to_select=5)
            rfe.fit(self.X_train, self.y_train)
            self.importances['rfe'] = rfe.ranking_
        except Exception:
            self.importances['rfe'] = [np.nan] * len(self.feature_names)

        # permutation importance
        try:
            perm = permutation_importance(self.model, self.X_test, self.y_test)
            self.importances['permutation'] = perm.importances_mean
        except Exception:
            self.importances['permutation'] = [np.nan] * len(self.feature_names)

        # gain-based (model-specific)
        try:
            if self.model_type == 'xgboost':
                gain_importance = self.model.get_booster().get_score(importance_type='gain')
                self.importances['gain'] = [gain_importance.get(f, 0) for f in self.feature_names]
            elif self.model_type == 'lightgbm':
                self.importances['gain'] = self.model.booster_.feature_importance(importance_type='gain')
            elif self.model_type == 'catboost':
                self.importances['gain'] = self.model.get_feature_importance(type='PredictionValuesChange')
        except Exception:
            self.importances['gain'] = [np.nan] * len(self.feature_names)

    def save_results(self, model_log_path='model_results.xlsx', importance_log_path='feature_importances.xlsx'):
        '''
        Saves metrics, params, and feature importance tables to Excel files.
        Appends to existing files if they exist.

        Parameters:
            model_log_path (str): Path to the Excel file for run-level metrics/params.
            importance_log_path (str): Path to the Excel file for feature importances.
                                       If "", importance logging is skipped.

        Returns:
            None
        '''
        # compile run-level record
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

        # append or create
        if os.path.exists(model_log_path):
            old = pd.read_excel(model_log_path)
            df_metrics = pd.concat([old, df_metrics], ignore_index=True)
        df_metrics.to_excel(model_log_path, index=False)

        # feature importances
        if importance_log_path != "":
            importance_df = pd.DataFrame(
                {name: values for name, values in self.importances.items()},
                index=self.feature_names
            ).reset_index()

            importance_df.insert(1, 'date', self.date)
            importance_df.insert(2, 'time', self.time)
            importance_df.insert(3, 'model_type', self.model_type)
            importance_df.insert(4, 'notes', self.notes)

            if os.path.exists(importance_log_path):
                old = pd.read_excel(importance_log_path)
                importance_df = pd.concat([old, importance_df], ignore_index=True)
            importance_df.to_excel(importance_log_path, index=False)
