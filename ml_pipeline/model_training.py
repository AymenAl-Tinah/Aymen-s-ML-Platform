"""
Model Training Module
=====================
Handles model selection, training, hyperparameter tuning, and cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import make_scorer, accuracy_score, f1_score, r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import logging
import joblib
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training system with automatic model selection,
    hyperparameter tuning, and cross-validation.
    """
    
    # Model registry with difficulty levels
    CLASSIFICATION_MODELS = {
        'basic': {
            'Logistic Regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [1000]
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            }
        },
        'intermediate': {
            'Random Forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'SVM': {
                'model': SVC,
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            }
        },
        'advanced': {
            'XGBoost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'eval_metric': ['logloss']
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'num_leaves': [15, 31, 63],
                    'subsample': [0.6, 0.8, 1.0],
                    'verbose': [-1]
                }
            }
        }
    }
    
    REGRESSION_MODELS = {
        'basic': {
            'Linear Regression': {
                'model': LinearRegression,
                'params': {
                    'fit_intercept': [True, False]
                }
            },
            'Decision Tree': {
                'model': DecisionTreeRegressor,
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        },
        'intermediate': {
            'Random Forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'Ridge Regression': {
                'model': Ridge,
                'params': {
                    'alpha': [0.01, 0.1, 1, 10, 100],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                }
            },
            'Lasso Regression': {
                'model': Lasso,
                'params': {
                    'alpha': [0.01, 0.1, 1, 10, 100],
                    'max_iter': [1000, 2000]
                }
            },
            'SVR': {
                'model': SVR,
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            }
        },
        'advanced': {
            'XGBoost': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'num_leaves': [15, 31, 63],
                    'subsample': [0.6, 0.8, 1.0],
                    'verbose': [-1]
                }
            }
        }
    }
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize ModelTrainer.
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.training_history = []
        
        # Select model registry
        if task_type == 'classification':
            self.model_registry = self.CLASSIFICATION_MODELS
        else:
            self.model_registry = self.REGRESSION_MODELS
        
        logger.info(f"ModelTrainer initialized for {task_type}")
    
    def prepare_data(self, X: pd.DataFrame, y: np.ndarray, 
                    test_size: float = 0.2, random_state: int = 42):
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature DataFrame
            y: Target array
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self.task_type == 'classification' and len(np.unique(y)) > 1 else None
        )
        
        logger.info(f"Data split: Train={self.X_train.shape}, Test={self.X_test.shape}")
    
    def train_model(self, model_name: str, level: str = 'basic', 
                   use_randomized_search: bool = True, 
                   n_iter: int = 20, cv: int = 5,
                   random_state: int = 42) -> Dict:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            level: Difficulty level ('basic', 'intermediate', 'advanced')
            use_randomized_search: Whether to use RandomizedSearchCV
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds
            random_state: Random seed
            
        Returns:
            Dictionary with training results
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # Get model configuration
        if level not in self.model_registry:
            raise ValueError(f"Invalid level: {level}")
        
        if model_name not in self.model_registry[level]:
            raise ValueError(f"Model '{model_name}' not found in {level} level")
        
        model_config = self.model_registry[level][model_name]
        model_class = model_config['model']
        param_grid = model_config['params']
        
        logger.info(f"Training {model_name} ({level})...")
        start_time = time.time()
        
        try:
            if use_randomized_search and len(param_grid) > 0:
                # Hyperparameter tuning with RandomizedSearchCV
                base_model = model_class(random_state=random_state)
                
                # Define scoring metric
                if self.task_type == 'classification':
                    scoring = 'accuracy'
                else:
                    scoring = 'r2'
                
                random_search = RandomizedSearchCV(
                    base_model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=0
                )
                
                random_search.fit(self.X_train, self.y_train)
                best_model = random_search.best_estimator_
                best_params = random_search.best_params_
                cv_score = random_search.best_score_
                
                logger.info(f"Best params: {best_params}")
                logger.info(f"CV Score: {cv_score:.4f}")
            else:
                # Train with default parameters
                best_model = model_class(random_state=random_state)
                best_model.fit(self.X_train, self.y_train)
                best_params = {}
                
                # Calculate CV score
                if self.task_type == 'classification':
                    cv_scores = cross_val_score(best_model, self.X_train, self.y_train, 
                                               cv=cv, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(best_model, self.X_train, self.y_train, 
                                               cv=cv, scoring='r2')
                cv_score = cv_scores.mean()
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Store trained model
            self.trained_models[model_name] = {
                'model': best_model,
                'level': level,
                'params': best_params,
                'cv_score': cv_score,
                'training_time': training_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update best model
            if cv_score > self.best_score:
                self.best_score = cv_score
                self.best_model = best_model
                self.best_model_name = model_name
            
            # Log training history
            self.training_history.append({
                'model_name': model_name,
                'level': level,
                'cv_score': cv_score,
                'training_time': training_time
            })
            
            logger.info(f"{model_name} trained successfully in {training_time:.2f}s")
            
            return {
                'model_name': model_name,
                'level': level,
                'cv_score': cv_score,
                'best_params': best_params,
                'training_time': training_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'level': level,
                'error': str(e),
                'success': False
            }
    
    def train_multiple_models(self, levels: List[str] = None, 
                            model_names: List[str] = None,
                            **kwargs) -> List[Dict]:
        """
        Train multiple models.
        
        Args:
            levels: List of difficulty levels to include
            model_names: Specific model names to train (overrides levels)
            **kwargs: Additional arguments for train_model()
            
        Returns:
            List of training results
        """
        results = []
        
        if model_names:
            # Train specific models
            for model_name in model_names:
                # Find the level for this model
                for level in self.model_registry:
                    if model_name in self.model_registry[level]:
                        result = self.train_model(model_name, level, **kwargs)
                        results.append(result)
                        break
        else:
            # Train all models in specified levels
            if levels is None:
                levels = ['basic', 'intermediate', 'advanced']
            
            for level in levels:
                if level in self.model_registry:
                    for model_name in self.model_registry[level]:
                        result = self.train_model(model_name, level, **kwargs)
                        results.append(result)
        
        logger.info(f"Training complete: {len(results)} models trained")
        return results
    
    def get_available_models(self, level: str = None) -> Dict:
        """
        Get list of available models.
        
        Args:
            level: Specific level to query (None for all)
            
        Returns:
            Dictionary of available models
        """
        if level:
            return {level: list(self.model_registry[level].keys())}
        else:
            return {
                level: list(models.keys()) 
                for level, models in self.model_registry.items()
            }
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        model_data = self.trained_models[model_name]
        joblib.dump(model_data, filepath)
        logger.info(f"Model '{model_name}' saved to {filepath}")
    
    def save_best_model(self, filepath: str):
        """
        Save the best performing model.
        
        Args:
            filepath: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No models trained yet")
        
        best_model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'score': self.best_score,
            'task_type': self.task_type,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(best_model_data, filepath)
        logger.info(f"Best model '{self.best_model_name}' saved to {filepath}")
    
    def save_all_models(self, directory: str):
        """
        Save all trained models to a directory.
        
        Args:
            directory: Directory to save models
        """
        import os
        os.makedirs(directory, exist_ok=True)
        
        for model_name in self.trained_models:
            safe_name = model_name.replace(' ', '_').lower()
            filepath = os.path.join(directory, f"{safe_name}.pkl")
            self.save_model(model_name, filepath)
        
        logger.info(f"All models saved to {directory}")
    
    def get_training_summary(self) -> Dict:
        """
        Get summary of all training activities.
        
        Returns:
            Dictionary with training summary
        """
        summary = {
            'task_type': self.task_type,
            'total_models_trained': len(self.trained_models),
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'training_history': self.training_history,
            'models': {}
        }
        
        for model_name, model_data in self.trained_models.items():
            summary['models'][model_name] = {
                'level': model_data['level'],
                'cv_score': model_data['cv_score'],
                'training_time': model_data['training_time'],
                'params': model_data['params']
            }
        
        return summary
