"""
Data Preprocessing Module
==========================
Handles missing values, encoding, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline with automatic handling
    of missing values, encoding, and scaling.
    """
    
    def __init__(self):
        self.df = None
        self.original_df = None
        self.feature_cols = []
        self.target_col = None
        self.numerical_cols = []
        self.categorical_cols = []
        
        # Transformers
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
        # Transformation log
        self.transformation_log = []
        
        # Processed data
        self.X = None
        self.y = None
        
    def set_data(self, df: pd.DataFrame, feature_cols: List[str], 
                 target_col: str, numerical_cols: List[str], 
                 categorical_cols: List[str]):
        """
        Set the dataset and column information.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            target_col: Target column name
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.numerical_cols = [col for col in numerical_cols if col in feature_cols]
        self.categorical_cols = [col for col in categorical_cols if col in feature_cols]
        
        logger.info("Data set for preprocessing:")
        logger.info(f"  Features: {len(self.feature_cols)}")
        logger.info(f"  Numerical: {len(self.numerical_cols)}")
        logger.info(f"  Categorical: {len(self.categorical_cols)}")
        logger.info(f"  Target: {self.target_col}")
    
    def handle_missing_values(self, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: Dictionary mapping column types to imputation strategies
                     {'numerical': 'mean'|'median'|'mode'|'drop',
                      'categorical': 'mode'|'constant'|'drop'}
                     
        Returns:
            DataFrame with missing values handled
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        # Default strategies
        if strategy is None:
            strategy = {
                'numerical': 'mean',
                'categorical': 'mode'
            }
        
        initial_shape = self.df.shape
        missing_before = self.df.isnull().sum().sum()
        
        # Handle numerical columns
        if self.numerical_cols and strategy.get('numerical') != 'drop':
            num_strategy = strategy.get('numerical', 'mean')
            if num_strategy in ['mean', 'median']:
                imputer = SimpleImputer(strategy=num_strategy)
                self.df[self.numerical_cols] = imputer.fit_transform(
                    self.df[self.numerical_cols]
                )
                self.imputers['numerical'] = imputer
                self.transformation_log.append(
                    f"Imputed missing values in {len(self.numerical_cols)} numerical "
                    f"columns using {num_strategy} strategy"
                )
            elif num_strategy == 'mode':
                for col in self.numerical_cols:
                    mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 0
                    self.df[col].fillna(mode_val, inplace=True)
                self.transformation_log.append(
                    f"Imputed missing values in numerical columns using mode"
                )
        
        # Handle categorical columns
        if self.categorical_cols and strategy.get('categorical') != 'drop':
            cat_strategy = strategy.get('categorical', 'mode')
            if cat_strategy == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
                self.df[self.categorical_cols] = imputer.fit_transform(
                    self.df[self.categorical_cols]
                )
                self.imputers['categorical'] = imputer
                self.transformation_log.append(
                    f"Imputed missing values in {len(self.categorical_cols)} categorical "
                    f"columns using mode strategy"
                )
            elif cat_strategy == 'constant':
                for col in self.categorical_cols:
                    self.df[col].fillna('Unknown', inplace=True)
                self.transformation_log.append(
                    f"Imputed missing values in categorical columns with 'Unknown'"
                )
        
        # Drop rows with missing values if specified
        if 'drop' in strategy.values():
            self.df.dropna(inplace=True)
            self.transformation_log.append(
                f"Dropped rows with missing values. Shape changed from "
                f"{initial_shape} to {self.df.shape}"
            )
        
        missing_after = self.df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} → {missing_after}")
        
        return self.df
    
    def encode_categorical_features(self, method: str = 'auto', 
                                    max_categories: int = 10) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            method: Encoding method ('auto', 'onehot', 'label')
            max_categories: Maximum unique values for one-hot encoding
            
        Returns:
            DataFrame with encoded features
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        if not self.categorical_cols:
            logger.info("No categorical columns to encode")
            return self.df
        
        encoded_cols = []
        
        for col in self.categorical_cols:
            unique_count = self.df[col].nunique()
            
            # Decide encoding method
            if method == 'auto':
                use_onehot = unique_count <= max_categories
            elif method == 'onehot':
                use_onehot = True
            else:
                use_onehot = False
            
            if use_onehot and unique_count > 1:
                # One-hot encoding
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                encoded_cols.extend(dummies.columns.tolist())
                self.df.drop(col, axis=1, inplace=True)
                
                self.transformation_log.append(
                    f"One-hot encoded '{col}' ({unique_count} categories) → "
                    f"{len(dummies.columns)} binary features"
                )
                logger.info(f"One-hot encoded: {col}")
            else:
                # Label encoding
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
                encoded_cols.append(col)
                
                self.transformation_log.append(
                    f"Label encoded '{col}' ({unique_count} categories) → numeric labels"
                )
                logger.info(f"Label encoded: {col}")
        
        # Update feature columns list
        self.feature_cols = [col for col in self.df.columns if col != self.target_col]
        self.numerical_cols = self.feature_cols.copy()
        self.categorical_cols = []
        
        return self.df
    
    def scale_features(self, method: str = 'standard', 
                      columns: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            method: Scaling method ('standard', 'minmax', 'none')
            columns: Specific columns to scale (default: all numerical)
            
        Returns:
            DataFrame with scaled features
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        if method == 'none':
            logger.info("Skipping feature scaling")
            return self.df
        
        # Determine columns to scale
        if columns is None:
            cols_to_scale = [col for col in self.numerical_cols 
                           if col in self.df.columns and col != self.target_col]
        else:
            cols_to_scale = [col for col in columns if col in self.df.columns]
        
        if not cols_to_scale:
            logger.info("No columns to scale")
            return self.df
        
        # Apply scaling
        if method == 'standard':
            scaler = StandardScaler()
            self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
            self.scalers['standard'] = scaler
            self.transformation_log.append(
                f"Applied StandardScaler to {len(cols_to_scale)} features "
                f"(mean=0, std=1)"
            )
            logger.info(f"StandardScaler applied to {len(cols_to_scale)} columns")
            
        elif method == 'minmax':
            scaler = MinMaxScaler()
            self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
            self.scalers['minmax'] = scaler
            self.transformation_log.append(
                f"Applied MinMaxScaler to {len(cols_to_scale)} features "
                f"(range: 0-1)"
            )
            logger.info(f"MinMaxScaler applied to {len(cols_to_scale)} columns")
        
        return self.df
    
    def encode_target(self) -> np.ndarray:
        """
        Encode target variable if categorical.
        
        Returns:
            Encoded target array
        """
        if self.df is None or self.target_col is None:
            raise ValueError("No data or target column set")
        
        target_data = self.df[self.target_col]
        
        # Check if target is categorical
        if not pd.api.types.is_numeric_dtype(target_data):
            le = LabelEncoder()
            encoded_target = le.fit_transform(target_data.astype(str))
            self.encoders['target'] = le
            
            self.transformation_log.append(
                f"Label encoded target '{self.target_col}' "
                f"({target_data.nunique()} classes)"
            )
            logger.info(f"Target encoded: {self.target_col}")
            return encoded_target
        else:
            return target_data.values
    
    def remove_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numerical features.
        
        Args:
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        initial_shape = self.df.shape
        
        numerical_features = [col for col in self.numerical_cols 
                            if col in self.df.columns and col != self.target_col]
        
        if not numerical_features:
            logger.info("No numerical features for outlier removal")
            return self.df
        
        if method == 'iqr':
            for col in numerical_features:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers_before = len(self.df)
                self.df = self.df[
                    (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
                ]
                outliers_removed = outliers_before - len(self.df)
                
                if outliers_removed > 0:
                    logger.info(f"Removed {outliers_removed} outliers from '{col}'")
            
            self.transformation_log.append(
                f"Removed outliers using IQR method (threshold={threshold}). "
                f"Shape: {initial_shape} → {self.df.shape}"
            )
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(self.df[numerical_features]))
            self.df = self.df[(z_scores < threshold).all(axis=1)]
            
            self.transformation_log.append(
                f"Removed outliers using Z-score method (threshold={threshold}). "
                f"Shape: {initial_shape} → {self.df.shape}"
            )
        
        logger.info(f"Outlier removal: {initial_shape} → {self.df.shape}")
        return self.df
    
    def prepare_train_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare final training data (X, y).
        
        Returns:
            Tuple of (features DataFrame, target array)
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        # Separate features and target
        self.X = self.df[[col for col in self.df.columns if col != self.target_col]]
        self.y = self.encode_target()
        
        logger.info(f"Training data prepared: X{self.X.shape}, y{self.y.shape}")
        
        return self.X, self.y
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of all preprocessing steps.
        
        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            'original_shape': self.original_df.shape if self.original_df is not None else None,
            'final_shape': self.df.shape if self.df is not None else None,
            'transformations': self.transformation_log,
            'numerical_features': len(self.numerical_cols),
            'categorical_features': len(self.categorical_cols),
            'total_features': len(self.feature_cols),
            'target_column': self.target_col,
            'scalers_used': list(self.scalers.keys()),
            'encoders_used': list(self.encoders.keys())
        }
        
        return summary
    
    def auto_preprocess(self, missing_strategy: Dict = None, 
                       encoding_method: str = 'auto',
                       scaling_method: str = 'standard',
                       remove_outliers: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Perform complete automatic preprocessing pipeline.
        
        Args:
            missing_strategy: Strategy for handling missing values
            encoding_method: Method for encoding categorical features
            scaling_method: Method for scaling features
            remove_outliers: Whether to remove outliers
            
        Returns:
            Tuple of (features DataFrame, target array)
        """
        logger.info("Starting automatic preprocessing pipeline...")
        
        # Step 1: Handle missing values
        self.handle_missing_values(missing_strategy)
        
        # Step 2: Remove outliers (optional)
        if remove_outliers:
            self.remove_outliers()
        
        # Step 3: Encode categorical features
        self.encode_categorical_features(method=encoding_method)
        
        # Step 4: Scale features
        self.scale_features(method=scaling_method)
        
        # Step 5: Prepare training data
        X, y = self.prepare_train_data()
        
        logger.info("Preprocessing pipeline complete!")
        logger.info(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def export_transformers(self, filepath: str):
        """
        Export fitted transformers for later use.
        
        Args:
            filepath: Path to save transformers
        """
        import joblib
        
        transformers = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'feature_cols': self.feature_cols,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols
        }
        
        joblib.dump(transformers, filepath)
        logger.info(f"Transformers exported to: {filepath}")
