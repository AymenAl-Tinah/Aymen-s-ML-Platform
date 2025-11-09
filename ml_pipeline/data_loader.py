"""
Data Loader Module
==================
Handles dataset loading, column detection, and initial analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Automatically loads and analyzes datasets, detecting feature types,
    target columns, and metadata columns.
    """
    
    def __init__(self):
        self.df = None
        self.original_df = None
        self.file_path = None
        self.column_types = {}
        self.detected_target = None
        self.detected_features = []
        self.detected_meta = []
        self.detection_justifications = {}
        
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        try:
            self.file_path = file_path
            self.df = pd.read_csv(file_path, **kwargs)
            self.original_df = self.df.copy()
            logger.info(f"Successfully loaded dataset: {file_path}")
            logger.info(f"Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_basic_info(self) -> Dict:
        """
        Get basic information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        return info
    
    def detect_column_types(self) -> Dict[str, List[str]]:
        """
        Automatically detect numerical and categorical columns.
        
        Returns:
            Dictionary with 'numerical' and 'categorical' column lists
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        numerical_cols = []
        categorical_cols = []
        
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check if it's actually categorical (few unique values)
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio < 0.05 and self.df[col].nunique() < 20:
                    categorical_cols.append(col)
                    self.detection_justifications[col] = (
                        f"Detected as categorical: {self.df[col].nunique()} unique values "
                        f"({unique_ratio*100:.1f}% of total rows)"
                    )
                else:
                    numerical_cols.append(col)
                    self.detection_justifications[col] = (
                        f"Detected as numerical: continuous values with "
                        f"{self.df[col].nunique()} unique values"
                    )
            else:
                categorical_cols.append(col)
                self.detection_justifications[col] = (
                    f"Detected as categorical: non-numeric dtype ({self.df[col].dtype}), "
                    f"{self.df[col].nunique()} unique values"
                )
        
        self.column_types = {
            'numerical': numerical_cols,
            'categorical': categorical_cols
        }
        
        return self.column_types
    
    def detect_target_column(self) -> Optional[str]:
        """
        Automatically detect the most likely target column.
        
        Strategy:
        1. Look for common target column names
        2. Find columns with binary or few unique values
        3. Prefer columns at the end of the dataset
        
        Returns:
            Name of detected target column
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Common target column names
        target_keywords = ['target', 'label', 'class', 'output', 'y', 'outcome', 
                          'result', 'prediction', 'response', 'dependent']
        
        # Check for exact or partial matches
        for col in self.df.columns:
            col_lower = col.lower()
            for keyword in target_keywords:
                if keyword in col_lower:
                    self.detected_target = col
                    self.detection_justifications[f"{col}_target"] = (
                        f"Detected '{col}' as target: column name contains keyword '{keyword}'"
                    )
                    logger.info(f"Target detected: {col} (keyword match)")
                    return self.detected_target
        
        # Look for binary or low-cardinality columns (likely classification targets)
        candidates = []
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            if 2 <= unique_count <= 20:  # Reasonable range for classification
                unique_ratio = unique_count / len(self.df)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    candidates.append((col, unique_count, unique_ratio))
        
        if candidates:
            # Prefer the last column with suitable characteristics
            candidates.sort(key=lambda x: (self.df.columns.get_loc(x[0]), -x[1]))
            self.detected_target = candidates[-1][0]
            self.detection_justifications[f"{self.detected_target}_target"] = (
                f"Detected '{self.detected_target}' as target: "
                f"{candidates[-1][1]} unique values ({candidates[-1][2]*100:.1f}% of rows), "
                f"positioned near end of dataset"
            )
            logger.info(f"Target detected: {self.detected_target} (heuristic)")
            return self.detected_target
        
        # Default to last column
        self.detected_target = self.df.columns[-1]
        self.detection_justifications[f"{self.detected_target}_target"] = (
            f"Detected '{self.detected_target}' as target: default to last column"
        )
        logger.info(f"Target detected: {self.detected_target} (default: last column)")
        return self.detected_target
    
    def detect_meta_columns(self) -> List[str]:
        """
        Detect metadata columns (IDs, timestamps, etc.) that should not be used as features.
        
        Returns:
            List of metadata column names
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        meta_cols = []
        meta_keywords = ['id', 'index', 'key', 'uuid', 'timestamp', 'date', 'time', 
                        'created', 'updated', 'modified', 'name', 'description']
        
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Check for keyword matches
            for keyword in meta_keywords:
                if keyword in col_lower:
                    meta_cols.append(col)
                    self.detection_justifications[f"{col}_meta"] = (
                        f"Detected '{col}' as metadata: column name contains '{keyword}'"
                    )
                    break
            
            # Check if column has all unique values (likely an ID)
            if col not in meta_cols and self.df[col].nunique() == len(self.df):
                meta_cols.append(col)
                self.detection_justifications[f"{col}_meta"] = (
                    f"Detected '{col}' as metadata: all values are unique (likely ID column)"
                )
        
        self.detected_meta = meta_cols
        logger.info(f"Metadata columns detected: {meta_cols}")
        return meta_cols
    
    def auto_detect_all(self) -> Dict:
        """
        Perform complete automatic detection of all column roles.
        
        Returns:
            Dictionary with detection results and justifications
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Detect column types
        self.detect_column_types()
        
        # Detect target
        self.detect_target_column()
        
        # Detect metadata
        self.detect_meta_columns()
        
        # Determine feature columns (exclude target and meta)
        all_cols = set(self.df.columns)
        exclude_cols = set([self.detected_target] + self.detected_meta)
        self.detected_features = list(all_cols - exclude_cols)
        
        detection_summary = {
            'target': self.detected_target,
            'features': self.detected_features,
            'meta': self.detected_meta,
            'column_types': self.column_types,
            'justifications': self.detection_justifications
        }
        
        logger.info("Auto-detection complete:")
        logger.info(f"  Target: {self.detected_target}")
        logger.info(f"  Features: {len(self.detected_features)} columns")
        logger.info(f"  Metadata: {len(self.detected_meta)} columns")
        
        return detection_summary
    
    def set_target_column(self, target_col: str):
        """
        Manually set the target column.
        
        Args:
            target_col: Name of target column
        """
        if target_col not in self.df.columns:
            raise ValueError(f"Column '{target_col}' not found in dataset")
        
        self.detected_target = target_col
        # Update features list
        all_cols = set(self.df.columns)
        exclude_cols = set([self.detected_target] + self.detected_meta)
        self.detected_features = list(all_cols - exclude_cols)
        logger.info(f"Target column manually set to: {target_col}")
    
    def set_feature_columns(self, feature_cols: List[str]):
        """
        Manually set feature columns.
        
        Args:
            feature_cols: List of feature column names
        """
        invalid_cols = [col for col in feature_cols if col not in self.df.columns]
        if invalid_cols:
            raise ValueError(f"Columns not found in dataset: {invalid_cols}")
        
        self.detected_features = feature_cols
        logger.info(f"Feature columns manually set: {len(feature_cols)} columns")
    
    def get_column_statistics(self, column: str) -> Dict:
        """
        Get detailed statistics for a specific column.
        
        Args:
            column: Column name
            
        Returns:
            Dictionary with column statistics
        """
        if self.df is None or column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        col_data = self.df[column]
        stats = {
            'name': column,
            'dtype': str(col_data.dtype),
            'count': len(col_data),
            'missing': col_data.isnull().sum(),
            'missing_pct': (col_data.isnull().sum() / len(col_data)) * 100,
            'unique': col_data.nunique(),
            'unique_pct': (col_data.nunique() / len(col_data)) * 100
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            stats.update({
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'median': col_data.median(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75)
            })
        else:
            value_counts = col_data.value_counts()
            stats.update({
                'top_values': value_counts.head(10).to_dict(),
                'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_freq': value_counts.iloc[0] if len(value_counts) > 0 else 0
            })
        
        return stats
