"""
Utilities Module
================
Helper functions and utilities for the ML pipeline.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Utils:
    """
    Utility functions for file management, configuration, and helpers.
    """
    
    @staticmethod
    def ensure_directory(directory: str):
        """
        Ensure a directory exists, create if it doesn't.
        
        Args:
            directory: Directory path
        """
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")
    
    @staticmethod
    def save_json(data: Dict, filepath: str):
        """
        Save dictionary to JSON file.
        
        Args:
            data: Dictionary to save
            filepath: Path to save file
        """
        Utils.ensure_directory(os.path.dirname(filepath))
        
        # Convert numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        converted_data = convert_types(data)
        
        with open(filepath, 'w') as f:
            json.dump(converted_data, f, indent=4)
        
        logger.info(f"JSON saved to: {filepath}")
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """
        Load dictionary from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded dictionary
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"JSON loaded from: {filepath}")
        return data
    
    @staticmethod
    def save_model(model: Any, filepath: str, metadata: Dict = None):
        """
        Save a model with optional metadata.
        
        Args:
            model: Model object to save
            filepath: Path to save file
            metadata: Optional metadata dictionary
        """
        Utils.ensure_directory(os.path.dirname(filepath))
        
        save_data = {
            'model': model,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to: {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> tuple:
        """
        Load a model with metadata.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Tuple of (model, metadata)
        """
        save_data = joblib.load(filepath)
        
        if isinstance(save_data, dict) and 'model' in save_data:
            model = save_data['model']
            metadata = save_data.get('metadata', {})
        else:
            # Old format or direct model save
            model = save_data
            metadata = {}
        
        logger.info(f"Model loaded from: {filepath}")
        return model, metadata
    
    @staticmethod
    def get_file_size(filepath: str) -> str:
        """
        Get human-readable file size.
        
        Args:
            filepath: Path to file
            
        Returns:
            Formatted file size string
        """
        size_bytes = os.path.getsize(filepath)
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.2f} TB"
    
    @staticmethod
    def list_files(directory: str, extension: str = None) -> List[str]:
        """
        List files in a directory.
        
        Args:
            directory: Directory path
            extension: Optional file extension filter (e.g., '.pkl')
            
        Returns:
            List of file paths
        """
        if not os.path.exists(directory):
            return []
        
        files = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if extension is None or filename.endswith(extension):
                    files.append(filepath)
        
        return files
    
    @staticmethod
    def create_project_config(config_data: Dict, filepath: str = 'project_config.json'):
        """
        Create or update project configuration file.
        
        Args:
            config_data: Configuration dictionary
            filepath: Path to config file
        """
        # Load existing config if it exists
        if os.path.exists(filepath):
            existing_config = Utils.load_json(filepath)
            existing_config.update(config_data)
            config_data = existing_config
        
        # Add timestamp
        config_data['last_updated'] = datetime.now().isoformat()
        
        Utils.save_json(config_data, filepath)
        logger.info("Project configuration updated")
    
    @staticmethod
    def load_project_config(filepath: str = 'project_config.json') -> Dict:
        """
        Load project configuration.
        
        Args:
            filepath: Path to config file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(filepath):
            logger.warning("Project config not found, returning empty config")
            return {}
        
        return Utils.load_json(filepath)
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict:
        """
        Validate a DataFrame and return validation results.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if empty
        if df.empty:
            validation['is_valid'] = False
            validation['errors'].append("DataFrame is empty")
            return validation
        
        # Check for all missing columns
        all_missing_cols = [col for col in df.columns if df[col].isnull().all()]
        if all_missing_cols:
            validation['warnings'].append(f"Columns with all missing values: {all_missing_cols}")
        
        # Check for high missing percentage
        high_missing = []
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                high_missing.append((col, missing_pct))
        
        if high_missing:
            validation['warnings'].append(
                f"Columns with >50% missing values: {[(col, f'{pct:.1f}%') for col, pct in high_missing]}"
            )
        
        # Check for single value columns
        single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
        if single_value_cols:
            validation['warnings'].append(f"Columns with single unique value: {single_value_cols}")
        
        return validation
    
    @staticmethod
    def detect_task_type(y: np.ndarray) -> str:
        """
        Automatically detect if task is classification or regression.
        
        Args:
            y: Target array
            
        Returns:
            'classification' or 'regression'
        """
        unique_values = len(np.unique(y))
        total_values = len(y)
        
        # If target has few unique values relative to total, it's likely classification
        if unique_values <= 20 or (unique_values / total_values) < 0.05:
            return 'classification'
        else:
            return 'regression'
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Format time in seconds to human-readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"
    
    @staticmethod
    def generate_unique_filename(base_name: str, extension: str, directory: str = '.') -> str:
        """
        Generate a unique filename by appending a number if file exists.
        
        Args:
            base_name: Base filename without extension
            extension: File extension (with or without dot)
            directory: Directory to check for existing files
            
        Returns:
            Unique filename
        """
        if not extension.startswith('.'):
            extension = '.' + extension
        
        filename = f"{base_name}{extension}"
        filepath = os.path.join(directory, filename)
        
        counter = 1
        while os.path.exists(filepath):
            filename = f"{base_name}_{counter}{extension}"
            filepath = os.path.join(directory, filename)
            counter += 1
        
        return filename
    
    @staticmethod
    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names by removing special characters and spaces.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned column names
        """
        df = df.copy()
        
        # Replace spaces and special characters
        df.columns = df.columns.str.replace('[^a-zA-Z0-9]', '_', regex=True)
        df.columns = df.columns.str.replace('_+', '_', regex=True)
        df.columns = df.columns.str.strip('_')
        df.columns = df.columns.str.lower()
        
        # Ensure unique column names
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [
                f"{dup}_{i}" if i != 0 else dup 
                for i in range(sum(cols == dup))
            ]
        df.columns = cols
        
        logger.info("Column names cleaned")
        return df
    
    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict:
        """
        Get detailed memory usage of a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with memory usage information
        """
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        usage_info = {
            'total_mb': total_memory / (1024 ** 2),
            'per_column': {
                col: mem / (1024 ** 2) 
                for col, mem in memory_usage.items()
            }
        }
        
        return usage_info
    
    @staticmethod
    def sample_dataframe(df: pd.DataFrame, n: int = 1000, 
                        stratify_col: str = None) -> pd.DataFrame:
        """
        Sample a DataFrame, optionally with stratification.
        
        Args:
            df: Input DataFrame
            n: Number of samples
            stratify_col: Column to stratify by
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= n:
            return df
        
        if stratify_col and stratify_col in df.columns:
            # Stratified sampling
            return df.groupby(stratify_col, group_keys=False).apply(
                lambda x: x.sample(min(len(x), n // df[stratify_col].nunique()))
            )
        else:
            # Random sampling
            return df.sample(n=n, random_state=42)
    
    @staticmethod
    def create_session_state() -> Dict:
        """
        Create a new session state dictionary.
        
        Returns:
            Session state dictionary
        """
        return {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'created_at': datetime.now().isoformat(),
            'data_loaded': False,
            'preprocessing_done': False,
            'training_done': False,
            'evaluation_done': False,
            'current_step': 'data_loading'
        }
    
    @staticmethod
    def update_session_state(state: Dict, updates: Dict) -> Dict:
        """
        Update session state with new values.
        
        Args:
            state: Current session state
            updates: Dictionary of updates
            
        Returns:
            Updated session state
        """
        state.update(updates)
        state['last_updated'] = datetime.now().isoformat()
        return state
