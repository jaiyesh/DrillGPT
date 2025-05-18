import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import logging
from colorama import Fore, Style

logger = logging.getLogger("DrillGPT")

class DrillingModels:
    """Predictive models for drilling optimization."""
    
    def __init__(self):
        """Initialize models."""
        self.rop_model = None
        self.bit_failure_model = None
        self.feature_importances = None
        self.formation_types = ['Sandstone', 'Shale', 'Limestone', 'Dolomite']
        self.bit_types = ['PDC', 'Tricone', 'Diamond']
        self.train_columns = None  # Store training columns for consistent prediction
    
    def preprocess_data(self, df, target):
        """Preprocess drilling data for modeling."""
        # Create dummy variables for categorical features
        # We need to ensure all categories are present, even if not in this particular dataset
        
        # For formation
        formation_dummies = pd.get_dummies(df['formation'], prefix='formation')
        # Add missing formation columns if any
        for formation in self.formation_types:
            col_name = f'formation_{formation}'
            if col_name not in formation_dummies.columns:
                formation_dummies[col_name] = 0
                
        # For bit type
        bit_dummies = pd.get_dummies(df['bit_type'], prefix='bit')
        # Add missing bit columns if any
        for bit_type in self.bit_types:
            col_name = f'bit_{bit_type}'
            if col_name not in bit_dummies.columns:
                bit_dummies[col_name] = 0
        
        # Combine with original data
        data = pd.concat([df.drop(['formation', 'bit_type'], axis=1), formation_dummies, bit_dummies], axis=1)
        
        # Remove target from features if not needed
        if target != 'bit_failure' and 'bit_failure' in data.columns:
            data = data.drop('bit_failure', axis=1)
        
        # Split into X and y
        X = data.drop(target, axis=1) if target in data.columns else data
        y = data[target] if target in data.columns else None
        
        # If this is the first processing, store column order
        if self.train_columns is None and y is not None:  # Only store during training
            self.train_columns = X.columns.tolist()
        elif self.train_columns is not None:
            # Ensure columns match training data
            missing_cols = set(self.train_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            # Ensure column order matches training data
            X = X[self.train_columns]
        
        return X, y
    
    def train_rop_model(self, df):
        """Train ROP prediction model."""
        logger.info(Fore.LIGHTBLUE_EX + "Training ROP prediction model..." + Style.RESET_ALL)
        X, y = self.preprocess_data(df, 'rop')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(Fore.LIGHTBLUE_EX + f"Training samples: {len(X_train)}, Test samples: {len(X_test)}" + Style.RESET_ALL)
        self.rop_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rop_model.fit(X_train, y_train)
        y_pred = self.rop_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(Fore.LIGHTGREEN_EX + f"ROP Model - MSE: {mse:.2f}, R²: {r2:.4f}" + Style.RESET_ALL)
        self.feature_importances = pd.Series(
            self.rop_model.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)
        return {'mse': mse, 'r2': r2, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}
    
    def train_bit_failure_model(self, df):
        """Train bit failure prediction model."""
        logger.info(Fore.LIGHTBLUE_EX + "Training bit failure prediction model..." + Style.RESET_ALL)
        X, y = self.preprocess_data(df, 'bit_failure')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logger.info(Fore.LIGHTBLUE_EX + f"Training samples: {len(X_train)}, Test samples: {len(X_test)}" + Style.RESET_ALL)
        self.bit_failure_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.bit_failure_model.fit(X_train, y_train)
        y_pred = self.bit_failure_model.predict(X_test)
        logger.info(Fore.LIGHTGREEN_EX + "Bit Failure Model Results:" + Style.RESET_ALL)
        logger.info(Fore.LIGHTGREEN_EX + "\n" + classification_report(y_test, y_pred) + Style.RESET_ALL)
        return {'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}
    
    def predict_rop(self, new_data):
        """Predict ROP for new data."""
        if self.rop_model is None:
            raise ValueError("ROP model not trained")
        
        X, _ = self.preprocess_data(new_data, 'rop')
        return self.rop_model.predict(X)
    
    def predict_bit_failure_prob(self, new_data):
        """Predict bit failure probability for new data."""
        if self.bit_failure_model is None:
            raise ValueError("Bit failure model not trained")
        
        X, _ = self.preprocess_data(new_data, 'bit_failure')
        return self.bit_failure_model.predict_proba(X)[:, 1] 