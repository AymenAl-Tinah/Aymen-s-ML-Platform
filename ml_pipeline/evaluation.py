"""
Model Evaluation Module
========================
Comprehensive model evaluation with metrics, visualizations, and comparisons.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, precision_recall_curve
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation system with metrics calculation,
    visualization, and model comparison.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize ModelEvaluator.
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.evaluation_results = {}
        
        logger.info(f"ModelEvaluator initialized for {task_type}")
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray = None,
                               model_name: str = 'Model') -> Dict:
        """
        Evaluate a classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # ROC-AUC (only for binary or with probabilities)
        try:
            if y_pred_proba is not None:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovr', average='weighted')
            else:
                metrics['roc_auc'] = None
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            metrics['roc_auc'] = None
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report
        try:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
            metrics['classification_report'] = None
        
        self.evaluation_results[model_name] = metrics
        
        logger.info(f"Classification evaluation for {model_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = 'Model') -> Dict:
        """
        Evaluate a regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Additional metrics
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        self.evaluation_results[model_name] = metrics
        
        logger.info(f"Regression evaluation for {model_name}:")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        
        return metrics
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: np.ndarray,
                      model_name: str = 'Model') -> Dict:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels/values
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        if self.task_type == 'classification':
            # Get prediction probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except:
                    pass
            
            return self.evaluate_classification(y_test, y_pred, y_pred_proba, model_name)
        else:
            return self.evaluate_regression(y_test, y_pred, model_name)
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet")
        
        comparison_data = []
        
        for model_name, metrics in self.evaluation_results.items():
            row = {'Model': model_name}
            
            if self.task_type == 'classification':
                row.update({
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'ROC-AUC': metrics.get('roc_auc', None)
                })
            else:
                row.update({
                    'MAE': metrics['mae'],
                    'MSE': metrics['mse'],
                    'RMSE': metrics['rmse'],
                    'R²': metrics['r2'],
                    'MAPE': metrics['mape']
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        if self.task_type == 'classification':
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        else:
            comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        return comparison_df
    
    def plot_confusion_matrix(self, model_name: str, 
                             class_names: List[str] = None) -> str:
        """
        Plot confusion matrix for a classification model.
        
        Args:
            model_name: Name of the model
            class_names: List of class names
            
        Returns:
            Plotly JSON string
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model '{model_name}' not found in evaluation results")
        
        if self.task_type != 'classification':
            raise ValueError("Confusion matrix only available for classification")
        
        cm = np.array(self.evaluation_results[model_name]['confusion_matrix'])
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorbar=dict(title="Count")
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            template='plotly_white',
            height=500,
            width=500
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def plot_model_comparison(self) -> str:
        """
        Create a bar chart comparing all models.
        
        Returns:
            Plotly JSON string
        """
        comparison_df = self.compare_models()
        
        if self.task_type == 'classification':
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metrics = [m for m in metrics if m in comparison_df.columns]
        else:
            metrics = ['R²', 'RMSE', 'MAE']
            metrics = [m for m in metrics if m in comparison_df.columns]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=len(metrics),
            subplot_titles=metrics,
            horizontal_spacing=0.1
        )
        
        for idx, metric in enumerate(metrics):
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    name=metric,
                    text=comparison_df[metric].round(4),
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=idx+1
            )
        
        fig.update_layout(
            title_text='Model Performance Comparison',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = 'Model') -> str:
        """
        Plot ROC curve for binary classification.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Plotly JSON string
        """
        if len(np.unique(y_true)) != 2:
            raise ValueError("ROC curve only available for binary classification")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc:.3f})',
            line=dict(color='darkorange', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_white',
            height=500,
            width=600,
            showlegend=True
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def plot_regression_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   model_name: str = 'Model') -> str:
        """
        Plot actual vs predicted values for regression.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Plotly JSON string
        """
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='steelblue', size=8, opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Calculate R²
        r2 = r2_score(y_true, y_pred)
        
        fig.update_layout(
            title=f'Actual vs Predicted - {model_name} (R² = {r2:.4f})',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            template='plotly_white',
            height=500,
            width=600,
            showlegend=True
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str = 'Model') -> str:
        """
        Plot residuals for regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Plotly JSON string
        """
        residuals = y_true - y_pred
        
        fig = go.Figure()
        
        # Residual plot
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='steelblue', size=8, opacity=0.6)
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f'Residual Plot - {model_name}',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            template='plotly_white',
            height=500,
            width=600
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def get_best_model(self) -> Tuple[str, Dict]:
        """
        Get the best performing model.
        
        Returns:
            Tuple of (model_name, metrics)
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet")
        
        if self.task_type == 'classification':
            best_model = max(self.evaluation_results.items(), 
                           key=lambda x: x[1]['accuracy'])
        else:
            best_model = max(self.evaluation_results.items(), 
                           key=lambda x: x[1]['r2'])
        
        return best_model
    
    def generate_evaluation_report(self) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Dictionary with complete evaluation report
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet")
        
        comparison_df = self.compare_models()
        best_model_name, best_metrics = self.get_best_model()
        
        report = {
            'task_type': self.task_type,
            'total_models': len(self.evaluation_results),
            'best_model': best_model_name,
            'best_metrics': best_metrics,
            'comparison_table': comparison_df.to_dict('records'),
            'all_results': self.evaluation_results
        }
        
        return report
