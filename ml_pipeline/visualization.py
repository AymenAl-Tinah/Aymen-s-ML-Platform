"""
Data Visualization Module
==========================
Creates interactive visualizations for data exploration and analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging
import json
import base64
from io import BytesIO
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class DataVisualizer:
    """
    Comprehensive data visualization toolkit for exploratory data analysis.
    """
    
    def __init__(self):
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.target_col = None
        
    def set_data(self, df: pd.DataFrame, numerical_cols: List[str] = None,
                 categorical_cols: List[str] = None, target_col: str = None):
        """
        Set the dataset for visualization.
        
        Args:
            df: Input DataFrame
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names
            target_col: Target column name
        """
        self.df = df.copy()
        self.numerical_cols = numerical_cols or []
        self.categorical_cols = categorical_cols or []
        self.target_col = target_col
        
        logger.info(f"Data set for visualization: {self.df.shape}")
    
    def plot_missing_values(self) -> str:
        """
        Create a visualization of missing values in the dataset.
        
        Returns:
            Plotly JSON string
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) == 0:
            # No missing values
            fig = go.Figure()
            fig.add_annotation(
                text="✓ No missing values found in the dataset!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=24, color="#10b981", family="Arial Black")
            )
        else:
            missing_pct = (missing_data / len(self.df)) * 100
            
            # Professional color gradient
            colors = px.colors.sequential.Sunset
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=missing_data.index,
                y=missing_pct.values,
                text=[f'{val:.1f}%' for val in missing_pct.values],
                textposition='auto',
                marker=dict(
                    color=missing_pct.values,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Missing %")
                ),
                hovertemplate='<b>%{x}</b><br>Missing: %{y:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(text='Missing Values by Column', font=dict(size=20, color='#1f2937')),
                xaxis_title='Columns',
                yaxis_title='Missing Percentage (%)',
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def plot_distribution(self, column: str = None) -> str:
        """
        Create distribution plots for numerical columns.
        
        Args:
            column: Specific column to plot (if None, plot all numerical)
            
        Returns:
            Plotly JSON string
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        cols_to_plot = [column] if column else self.numerical_cols[:6]  # Limit to 6
        
        if not cols_to_plot:
            fig = go.Figure()
            fig.add_annotation(
                text="No numerical columns to visualize",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create subplots
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=cols_to_plot,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for idx, col in enumerate(cols_to_plot):
            row = idx // n_cols + 1
            col_pos = idx % n_cols + 1
            
            # Professional color palette
            colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a']
            color = colors[idx % len(colors)]
            
            fig.add_trace(
                go.Histogram(
                    x=self.df[col],
                    name=col,
                    showlegend=False,
                    marker=dict(
                        color=color,
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                ),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title_text='Distribution of Numerical Features',
            template='plotly_white',
            height=300 * n_rows,
            showlegend=False
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def plot_correlation_matrix(self) -> str:
        """
        Create a correlation heatmap for numerical features.
        
        Returns:
            Plotly JSON string
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        if not self.numerical_cols:
            fig = go.Figure()
            fig.add_annotation(
                text="No numerical columns for correlation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Calculate correlation matrix
        corr_data = self.df[self.numerical_cols].corr()
        
        # Create heatmap with professional colors
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu_r',  # Reversed for better visualization
            zmid=0,
            text=corr_data.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10, "color": "white"},
            colorbar=dict(
                title="Correlation",
                tickmode='linear',
                tick0=-1,
                dtick=0.5
            ),
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            template='plotly_white',
            height=max(500, len(self.numerical_cols) * 30),
            width=max(500, len(self.numerical_cols) * 30)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def plot_categorical_distribution(self, max_categories: int = 10) -> str:
        """
        Create pie charts for categorical columns.
        
        Args:
            max_categories: Maximum number of categories to display
            
        Returns:
            Plotly JSON string
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        cols_to_plot = self.categorical_cols[:6]  # Limit to 6
        
        if not cols_to_plot:
            fig = go.Figure()
            fig.add_annotation(
                text="No categorical columns to visualize",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create subplots
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=cols_to_plot,
            specs=[[{'type': 'pie'} for _ in range(n_cols)] for _ in range(n_rows)],
            vertical_spacing=0.15
        )
        
        # Professional color palettes
        color_palettes = [
            px.colors.qualitative.Set3,
            px.colors.qualitative.Pastel,
            px.colors.qualitative.Bold,
            px.colors.qualitative.Vivid,
            px.colors.qualitative.Safe,
            px.colors.qualitative.Prism
        ]
        
        for idx, col in enumerate(cols_to_plot):
            row = idx // n_cols + 1
            col_pos = idx % n_cols + 1
            
            value_counts = self.df[col].value_counts().head(max_categories)
            colors = color_palettes[idx % len(color_palettes)]
            
            fig.add_trace(
                go.Pie(
                    labels=value_counts.index,
                    values=value_counts.values,
                    name=col,
                    showlegend=True,
                    marker=dict(colors=colors, line=dict(color='white', width=2)),
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                    textposition='inside',
                    textinfo='percent+label'
                ),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title_text='Distribution of Categorical Features',
            template='plotly_white',
            height=400 * n_rows
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def plot_target_distribution(self) -> str:
        """
        Visualize target variable distribution.
        
        Returns:
            Plotly JSON string
        """
        if self.df is None or self.target_col is None:
            raise ValueError("No data or target column set")
        
        target_data = self.df[self.target_col]
        
        # Check if target is numerical or categorical
        if pd.api.types.is_numeric_dtype(target_data) and target_data.nunique() > 20:
            # Numerical target - histogram with gradient
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=target_data,
                marker=dict(
                    color=target_data,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=self.target_col)
                ),
                nbinsx=30,
                hovertemplate='<b>Value: %{x}</b><br>Count: %{y}<extra></extra>'
            ))
            fig.update_layout(
                title=f'Distribution of Target: {self.target_col}',
                xaxis_title=self.target_col,
                yaxis_title='Frequency',
                template='plotly_white',
                height=500
            )
        else:
            # Categorical target - bar chart with gradient
            value_counts = target_data.value_counts()
            
            # Professional gradient colors
            colors = px.colors.sequential.Tealgrn
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                marker=dict(
                    color=value_counts.values,
                    colorscale='Tealgrn',
                    showscale=True,
                    colorbar=dict(title="Count"),
                    line=dict(color='white', width=1.5)
                ),
                text=value_counts.values,
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            ))
            fig.update_layout(
                title=f'Distribution of Target: {self.target_col}',
                xaxis_title=self.target_col,
                yaxis_title='Count',
                template='plotly_white',
                height=500
            )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def plot_outliers_boxplot(self) -> str:
        """
        Create box plots to visualize outliers in numerical features.
        
        Returns:
            Plotly JSON string
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        cols_to_plot = self.numerical_cols[:10]  # Limit to 10
        
        if not cols_to_plot:
            fig = go.Figure()
            fig.add_annotation(
                text="No numerical columns for outlier detection",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        fig = go.Figure()
        
        # Professional color palette for box plots
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', 
                  '#fa709a', '#fee140', '#30cfd0', '#a8edea', '#fed6e3']
        
        for idx, col in enumerate(cols_to_plot):
            color = colors[idx % len(colors)]
            fig.add_trace(go.Box(
                y=self.df[col],
                name=col,
                boxmean='sd',
                marker=dict(
                    color=color,
                    line=dict(color='#1f2937', width=1.5)
                ),
                line=dict(color='#1f2937', width=2),
                fillcolor=color,
                hovertemplate='<b>%{fullData.name}</b><br>Value: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Outlier Detection - Box Plots',
            yaxis_title='Values',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importances: np.ndarray, top_n: int = 20) -> str:
        """
        Plot feature importance from a trained model.
        
        Args:
            feature_names: List of feature names
            importances: Array of importance values
            top_n: Number of top features to display
            
        Returns:
            Plotly JSON string
        """
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color='steelblue',
            text=importance_df['importance'].round(4),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Features',
            template='plotly_white',
            height=max(400, top_n * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def create_pairplot_sample(self, n_features: int = 4) -> str:
        """
        Create a pair plot for a sample of features.
        
        Args:
            n_features: Number of features to include
            
        Returns:
            Base64 encoded image string
        """
        if self.df is None:
            raise ValueError("No data set. Please use set_data() first.")
        
        # Select features
        features = self.numerical_cols[:n_features]
        
        if len(features) < 2:
            return None
        
        # Create pair plot
        plot_df = self.df[features].sample(min(1000, len(self.df)))
        
        fig = sns.pairplot(plot_df, diag_kind='kde', plot_kws={'alpha': 0.6})
        fig.fig.suptitle('Pair Plot of Selected Features', y=1.02)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
    
    def generate_all_plots(self) -> Dict[str, str]:
        """
        Generate all available plots.
        
        Returns:
            Dictionary mapping plot names to their JSON/base64 representations
        """
        plots = {}
        
        try:
            plots['missing_values'] = self.plot_missing_values()
        except Exception as e:
            logger.error(f"Error creating missing values plot: {e}")
        
        try:
            plots['distribution'] = self.plot_distribution()
        except Exception as e:
            logger.error(f"Error creating distribution plot: {e}")
        
        try:
            plots['correlation'] = self.plot_correlation_matrix()
        except Exception as e:
            logger.error(f"Error creating correlation plot: {e}")
        
        try:
            plots['categorical'] = self.plot_categorical_distribution()
        except Exception as e:
            logger.error(f"Error creating categorical plot: {e}")
        
        try:
            if self.target_col:
                plots['target'] = self.plot_target_distribution()
        except Exception as e:
            logger.error(f"Error creating target plot: {e}")
        
        try:
            plots['outliers'] = self.plot_outliers_boxplot()
        except Exception as e:
            logger.error(f"Error creating outliers plot: {e}")
        
        logger.info(f"Generated {len(plots)} plots")
        return plots
