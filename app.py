"""
Aymen's ML Platform - Flask Application
========================================
Main Flask application for automated machine learning platform.

© 2025 Aymen's Labs - All Rights Reserved
"""

from flask import Flask, render_template, request, jsonify, session, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import traceback
import logging
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to convert numpy types to Python types
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle NaN and Inf
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, float):
        # Handle Python float NaN and Inf
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj

# Import ML pipeline modules
from ml_pipeline import (
    DataLoader, DataPreprocessor, DataVisualizer,
    ModelTrainer, ModelEvaluator, ReportGenerator, Utils
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'ml_auto_project_secret_key_2024'  # Change in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['MODELS_FOLDER'] = 'models'
app.config['REPORTS_FOLDER'] = 'reports'
app.config['CHARTS_FOLDER'] = 'charts'

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODELS_FOLDER'], app.config['REPORTS_FOLDER'], app.config['CHARTS_FOLDER']]:
    Utils.ensure_directory(folder)

# Global objects (in production, use proper session management or database)
data_loaders = {}
preprocessors = {}
visualizers = {}
trainers = {}
evaluators = {}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_id():
    """Get or create session ID."""
    if 'session_id' not in session:
        session['session_id'] = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return session['session_id']


def get_data_loader():
    """Get DataLoader for current session."""
    session_id = get_session_id()
    if session_id not in data_loaders:
        data_loaders[session_id] = DataLoader()
    return data_loaders[session_id]


def get_preprocessor():
    """Get DataPreprocessor for current session."""
    session_id = get_session_id()
    if session_id not in preprocessors:
        preprocessors[session_id] = DataPreprocessor()
    return preprocessors[session_id]


def get_visualizer():
    """Get DataVisualizer for current session."""
    session_id = get_session_id()
    if session_id not in visualizers:
        visualizers[session_id] = DataVisualizer()
    return visualizers[session_id]


def get_trainer():
    """Get ModelTrainer for current session."""
    session_id = get_session_id()
    if session_id not in trainers:
        trainers[session_id] = None  # Will be created when task type is known
    return trainers[session_id]


def get_evaluator():
    """Get ModelEvaluator for current session."""
    session_id = get_session_id()
    if session_id not in evaluators:
        evaluators[session_id] = None  # Will be created when task type is known
    return evaluators[session_id]


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main landing page."""
    return render_template('index.html')


@app.route('/data')
def data_page():
    """Data loading and exploration page."""
    return render_template('data.html')


@app.route('/preprocess')
def preprocess_page():
    """Data preprocessing page."""
    return render_template('preprocess.html')


@app.route('/visualize')
def visualize_page():
    """Data visualization page."""
    return render_template('visualize.html')


@app.route('/train')
def train_page():
    """Model training page."""
    return render_template('train.html')


@app.route('/evaluate')
def evaluate_page():
    """Model evaluation page."""
    return render_template('evaluate.html')


@app.route('/documentation')
def documentation_page():
    """Documentation and report generation page."""
    return render_template('documentation.html')


# ============================================================================
# API ENDPOINTS - DATA LOADING
# ============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and load dataset."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Only CSV files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{get_session_id()}_{filename}")
        file.save(filepath)
        
        # Load data
        loader = get_data_loader()
        df = loader.load_data(filepath)
        
        # Get basic info
        basic_info = loader.get_basic_info()
        
        # Store in session
        session['data_loaded'] = True
        session['filename'] = filename
        session['filepath'] = filepath
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'info': convert_to_serializable({
                'filename': filename,
                'shape': basic_info['shape'],
                'columns': basic_info['columns'],
                'missing_values': basic_info['missing_values'],
                'duplicate_rows': basic_info['duplicate_rows']
            })
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/detect_columns', methods=['POST'])
def detect_columns():
    """Auto-detect column types and roles."""
    try:
        loader = get_data_loader()
        
        if loader.df is None:
            return jsonify({'success': False, 'error': 'No data loaded'}), 400
        
        # Perform auto-detection
        detection_results = loader.auto_detect_all()
        
        # Store in session
        session['detection_results'] = detection_results
        
        return jsonify({
            'success': True,
            'detection': convert_to_serializable(detection_results)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/confirm_columns', methods=['POST'])
def confirm_columns():
    """Confirm or update column selections."""
    try:
        data = request.json
        loader = get_data_loader()
        
        if loader.df is None:
            return jsonify({'success': False, 'error': 'No data loaded'}), 400
        
        # Update selections
        if 'target' in data:
            loader.set_target_column(data['target'])
        
        if 'features' in data:
            loader.set_feature_columns(data['features'])
        
        # Store in session
        session['target_column'] = loader.detected_target
        session['feature_columns'] = loader.detected_features
        session['meta_columns'] = loader.detected_meta
        session['column_types'] = loader.column_types
        
        return jsonify({
            'success': True,
            'message': 'Column selections confirmed',
            'target': loader.detected_target,
            'features': loader.detected_features,
            'meta': loader.detected_meta
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/column_stats/<column_name>', methods=['GET'])
def get_column_stats(column_name):
    """Get statistics for a specific column."""
    try:
        loader = get_data_loader()
        
        if loader.df is None:
            return jsonify({'success': False, 'error': 'No data loaded'}), 400
        
        stats = loader.get_column_statistics(column_name)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - PREPROCESSING
# ============================================================================

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Preprocess the dataset."""
    try:
        data = request.json
        loader = get_data_loader()
        preprocessor = get_preprocessor()
        
        if loader.df is None:
            return jsonify({'success': False, 'error': 'No data loaded'}), 400
        
        # Get parameters
        missing_strategy = data.get('missing_strategy', {'numerical': 'mean', 'categorical': 'mode'})
        encoding_method = data.get('encoding_method', 'auto')
        scaling_method = data.get('scaling_method', 'standard')
        remove_outliers = data.get('remove_outliers', False)
        
        # Set data
        preprocessor.set_data(
            loader.df,
            session.get('feature_columns', []),
            session.get('target_column'),
            loader.column_types.get('numerical', []),
            loader.column_types.get('categorical', [])
        )
        
        # Perform preprocessing
        X, y = preprocessor.auto_preprocess(
            missing_strategy=missing_strategy,
            encoding_method=encoding_method,
            scaling_method=scaling_method,
            remove_outliers=remove_outliers
        )
        
        # Get summary
        summary = preprocessor.get_preprocessing_summary()
        
        # Store in session
        session['preprocessing_done'] = True
        session['preprocessing_summary'] = summary
        session['X_shape'] = X.shape
        session['y_shape'] = y.shape
        
        # Detect task type
        task_type = Utils.detect_task_type(y)
        session['task_type'] = task_type
        
        return jsonify({
            'success': True,
            'message': 'Preprocessing completed successfully',
            'summary': summary,
            'task_type': task_type,
            'X_shape': X.shape,
            'y_shape': y.shape
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# API ENDPOINTS - VISUALIZATION
# ============================================================================

@app.route('/api/visualize/all', methods=['GET'])
def visualize_all():
    """Generate all visualizations."""
    try:
        loader = get_data_loader()
        visualizer = get_visualizer()
        
        if loader.df is None:
            return jsonify({'success': False, 'error': 'No data loaded. Please upload a dataset first.'}), 400
        
        # Auto-detect column types if not already done
        if not loader.column_types or not loader.column_types.get('numerical') and not loader.column_types.get('categorical'):
            logger.info("Column types not detected, performing auto-detection...")
            loader.auto_detect_all()
        
        numerical_cols = loader.column_types.get('numerical', [])
        categorical_cols = loader.column_types.get('categorical', [])
        target_col = session.get('target_column') or loader.detected_target
        
        logger.info(f"Setting visualizer data - Numerical: {len(numerical_cols)}, Categorical: {len(categorical_cols)}, Target: {target_col}")
        
        # Set data
        visualizer.set_data(
            loader.df,
            numerical_cols,
            categorical_cols,
            target_col
        )
        
        # Generate all plots
        plots = visualizer.generate_all_plots()
        
        if not plots or len(plots) == 0:
            return jsonify({'success': False, 'error': 'No plots could be generated. Please ensure your data has numerical or categorical columns.'}), 400
        
        logger.info(f"Successfully generated {len(plots)} plots")
        
        # Save plots as PNG files in charts folder
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            
            for plot_type, plot_json in plots.items():
                try:
                    # Parse JSON and create figure
                    plot_data = json.loads(plot_json)
                    fig = go.Figure(data=plot_data['data'], layout=plot_data.get('layout', {}))
                    
                    # Save as PNG
                    chart_path = os.path.join(app.config['CHARTS_FOLDER'], f"{plot_type}.png")
                    fig.write_image(chart_path, format='png', width=1200, height=600, engine='kaleido')
                    logger.info(f"Saved chart: {chart_path}")
                except Exception as e:
                    logger.warning(f"Could not save {plot_type} as PNG: {str(e)}")
        except Exception as e:
            logger.warning(f"Could not save charts as PNG files: {str(e)}")
        
        return jsonify({
            'success': True,
            'plots': plots,
            'info': {
                'num_plots': len(plots),
                'numerical_cols': len(numerical_cols),
                'categorical_cols': len(categorical_cols)
            }
        })
    
    except Exception as e:
        logger.error(f"Error in visualize_all: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/visualize/<plot_type>', methods=['GET'])
def visualize_specific(plot_type):
    """Generate a specific visualization."""
    try:
        loader = get_data_loader()
        visualizer = get_visualizer()
        
        if loader.df is None:
            return jsonify({'success': False, 'error': 'No data loaded'}), 400
        
        # Set data
        visualizer.set_data(
            loader.df,
            loader.column_types.get('numerical', []),
            loader.column_types.get('categorical', []),
            session.get('target_column')
        )
        
        # Generate specific plot
        plot_methods = {
            'missing': visualizer.plot_missing_values,
            'distribution': visualizer.plot_distribution,
            'correlation': visualizer.plot_correlation_matrix,
            'categorical': visualizer.plot_categorical_distribution,
            'target': visualizer.plot_target_distribution,
            'outliers': visualizer.plot_outliers_boxplot
        }
        
        if plot_type not in plot_methods:
            return jsonify({'success': False, 'error': f'Unknown plot type: {plot_type}'}), 400
        
        plot_json = plot_methods[plot_type]()
        
        return jsonify({
            'success': True,
            'plot': plot_json
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - MODEL TRAINING
# ============================================================================

@app.route('/api/models/available', methods=['GET'])
def get_available_models():
    """Get list of available models."""
    try:
        task_type = session.get('task_type', 'classification')
        trainer = ModelTrainer(task_type=task_type)
        
        available_models = trainer.get_available_models()
        
        return jsonify({
            'success': True,
            'task_type': task_type,
            'models': available_models
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_models():
    """Train selected models."""
    try:
        data = request.json
        preprocessor = get_preprocessor()
        
        if preprocessor.X is None or preprocessor.y is None:
            return jsonify({'success': False, 'error': 'Data not preprocessed'}), 400
        
        # Get parameters
        task_type = session.get('task_type', 'classification')
        levels = data.get('levels', ['basic', 'intermediate', 'advanced'])
        model_names = data.get('model_names', None)
        test_size = data.get('test_size', 0.2)
        cv_folds = data.get('cv_folds', 5)
        n_iter = data.get('n_iter', 20)
        
        # Create trainer
        trainer = ModelTrainer(task_type=task_type)
        trainers[get_session_id()] = trainer
        
        # Prepare data
        trainer.prepare_data(preprocessor.X, preprocessor.y, test_size=test_size)
        
        # Train models
        results = trainer.train_multiple_models(
            levels=levels,
            model_names=model_names,
            n_iter=n_iter,
            cv=cv_folds
        )
        
        # Get training summary
        summary = trainer.get_training_summary()
        
        # Store in session
        session['training_done'] = True
        session['training_summary'] = summary
        
        # Save best model
        best_model_path = os.path.join(app.config['MODELS_FOLDER'], f"{get_session_id()}_best_model.pkl")
        trainer.save_best_model(best_model_path)
        
        return jsonify({
            'success': True,
            'message': 'Training completed successfully',
            'results': results,
            'summary': summary,
            'best_model': summary['best_model'],
            'best_score': summary['best_score']
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


# ============================================================================
# API ENDPOINTS - MODEL EVALUATION
# ============================================================================

@app.route('/api/evaluate', methods=['POST'])
def evaluate_models():
    """Evaluate all trained models."""
    try:
        trainer = get_trainer()
        
        if trainer is None or not trainer.trained_models:
            return jsonify({'success': False, 'error': 'No models trained'}), 400
        
        # Create evaluator
        task_type = session.get('task_type', 'classification')
        evaluator = ModelEvaluator(task_type=task_type)
        evaluators[get_session_id()] = evaluator
        
        # Evaluate all models
        for model_name, model_data in trainer.trained_models.items():
            evaluator.evaluate_model(
                model_data['model'],
                trainer.X_test,
                trainer.y_test,
                model_name
            )
        
        # Get comparison
        comparison_df = evaluator.compare_models()
        
        # Get best model
        best_model_name, best_metrics = evaluator.get_best_model()
        
        # Generate evaluation report
        eval_report = evaluator.generate_evaluation_report()
        
        # Store in session
        session['evaluation_done'] = True
        session['evaluation_report'] = eval_report
        
        return jsonify({
            'success': True,
            'message': 'Evaluation completed successfully',
            'comparison': convert_to_serializable(comparison_df.to_dict('records')),
            'best_model': best_model_name,
            'best_metrics': convert_to_serializable(best_metrics),
            'report': convert_to_serializable(eval_report)
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/evaluate/plot/<plot_type>', methods=['POST'])
def get_evaluation_plot(plot_type):
    """Get evaluation visualization."""
    try:
        evaluator = get_evaluator()
        
        if evaluator is None:
            return jsonify({'success': False, 'error': 'No evaluation performed'}), 400
        
        data = request.json
        model_name = data.get('model_name')
        
        if plot_type == 'comparison':
            plot_json = evaluator.plot_model_comparison()
        elif plot_type == 'confusion_matrix' and model_name:
            plot_json = evaluator.plot_confusion_matrix(model_name)
        else:
            return jsonify({'success': False, 'error': f'Unknown plot type: {plot_type}'}), 400
        
        return jsonify({
            'success': True,
            'plot': plot_json
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - REPORT GENERATION
# ============================================================================

@app.route('/api/report/generate', methods=['POST'])
def generate_report():
    """Generate comprehensive project report."""
    try:
        data = request.json
        
        # Gather all project data
        loader = get_data_loader()
        preprocessor = get_preprocessor()
        trainer = get_trainer()
        evaluator = get_evaluator()
        
        if loader.df is None:
            return jsonify({'success': False, 'error': 'No data loaded'}), 400
        
        # Prepare project data
        project_data = {
            'project_name': data.get('project_name', 'ML Auto Project'),
            'task_type': session.get('task_type', 'classification'),
            'include_visualizations': data.get('include_visualizations', True),
            'dataset_info': {
                'n_samples': int(loader.df.shape[0]),
                'n_features': int(loader.df.shape[1] - 1) if session.get('target_column') else int(loader.df.shape[1]),
                'target_column': session.get('target_column'),
                'numerical_features': loader.column_types.get('numerical', []),
                'categorical_features': loader.column_types.get('categorical', []),
                'missing_values': int(loader.df.isnull().sum().sum()),
                'duplicate_rows': int(loader.df.duplicated().sum())
            }
        }
        
        # Add preprocessing info
        if preprocessor and session.get('preprocessing_done'):
            project_data['preprocessing_info'] = session.get('preprocessing_summary', {})
        
        # Add visualizations if requested
        if data.get('include_visualizations', False):
            try:
                # Check for existing charts in charts folder
                chart_types = ['missing_values', 'distribution', 'correlation', 'categorical', 'target', 'outliers']
                visualizations = {}
                missing_charts = []
                
                # Check which charts exist
                for chart_type in chart_types:
                    chart_path = os.path.join(app.config['CHARTS_FOLDER'], f"{chart_type}.png")
                    if os.path.exists(chart_path):
                        visualizations[chart_type] = chart_path
                        logger.info(f"Found existing chart: {chart_path}")
                    else:
                        missing_charts.append(chart_type)
                        logger.info(f"Chart missing: {chart_type}")
                
                # If any charts are missing, generate them
                if missing_charts:
                    logger.info(f"Generating {len(missing_charts)} missing charts...")
                    visualizer = get_visualizer()
                    if visualizer and loader.df is not None:
                        # Set data in visualizer
                        numerical_cols = loader.column_types.get('numerical', [])
                        categorical_cols = loader.column_types.get('categorical', [])
                        target_col = session.get('target_column') or loader.detected_target
                        
                        visualizer.set_data(loader.df, numerical_cols, categorical_cols, target_col)
                        
                        # Generate plots
                        plots = visualizer.generate_all_plots()
                        
                        # Save missing charts
                        import plotly.graph_objects as go
                        import plotly.io as pio
                        
                        for chart_type in missing_charts:
                            if chart_type in plots:
                                try:
                                    plot_data = json.loads(plots[chart_type])
                                    fig = go.Figure(data=plot_data['data'], layout=plot_data.get('layout', {}))
                                    
                                    chart_path = os.path.join(app.config['CHARTS_FOLDER'], f"{chart_type}.png")
                                    fig.write_image(chart_path, format='png', width=1200, height=600, engine='kaleido')
                                    visualizations[chart_type] = chart_path
                                    logger.info(f"Generated and saved: {chart_path}")
                                except Exception as e:
                                    logger.error(f"Could not generate {chart_type}: {str(e)}")
                
                if visualizations:
                    project_data['visualizations'] = visualizations
                    logger.info(f"Added {len(visualizations)} visualizations to report")
                else:
                    logger.warning("No visualizations available for report")
                    
            except Exception as e:
                logger.error(f"Could not prepare visualizations: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Add model comparison
        if evaluator and session.get('evaluation_done'):
            comparison_df = evaluator.compare_models()
            project_data['comparison_df'] = comparison_df
            
            # Best model info
            best_model_name, best_metrics = evaluator.get_best_model()
            
            # Convert metrics to serializable format
            serializable_metrics = {}
            if best_metrics:
                for key, value in best_metrics.items():
                    if isinstance(value, (np.integer, np.floating)):
                        serializable_metrics[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        serializable_metrics[key] = value.tolist()
                    else:
                        serializable_metrics[key] = value
            
            project_data['best_model_info'] = {
                'model_name': best_model_name,
                'model_type': session.get('task_type'),
                'metrics': serializable_metrics,
                'hyperparameters': trainer.trained_models[best_model_name].get('params', {}) if best_model_name in trainer.trained_models else {},
                'training_time': float(trainer.trained_models[best_model_name].get('training_time', 0)) if best_model_name in trainer.trained_models else 0
            }
            project_data['best_model_name'] = best_model_name
            project_data['n_models'] = len(trainer.trained_models)
        
        # Generate report
        report_gen = ReportGenerator()
        report_filename = Utils.generate_unique_filename(
            f"ml_report_{get_session_id()}",
            'docx',
            app.config['REPORTS_FOLDER']
        )
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        
        report_gen.generate_complete_report(project_data, report_path)
        
        return jsonify({
            'success': True,
            'message': 'Report generated successfully',
            'filename': report_filename,
            'download_url': f'/api/report/download/{report_filename}'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/report/download/<filename>', methods=['GET'])
def download_report(filename):
    """Download generated report."""
    try:
        filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Report not found'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - MODEL SAVING
# ============================================================================

@app.route('/api/model/save', methods=['POST'])
def save_model():
    """Save a trained model with prediction code and README."""
    try:
        data = request.json
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'success': False, 'error': 'Model name is required'}), 400
        
        trainer = get_trainer()
        if trainer is None or not trainer.trained_models:
            return jsonify({'success': False, 'error': 'No models trained'}), 400
        
        if model_name not in trainer.trained_models:
            return jsonify({'success': False, 'error': f'Model {model_name} not found'}), 404
        
        # Get model data
        model_data = trainer.trained_models[model_name]
        model = model_data['model']
        
        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_model_name = model_name.replace(' ', '_').replace('/', '_')
        base_filename = f"{safe_model_name}_{timestamp}"
        
        # Save model
        model_path = os.path.join(app.config['MODELS_FOLDER'], f"{base_filename}.pkl")
        import joblib
        joblib.dump(model, model_path)
        
        # Generate prediction code
        prediction_code = generate_prediction_code(model_name, base_filename, trainer)
        code_path = os.path.join(app.config['MODELS_FOLDER'], f"{base_filename}_predict.py")
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(prediction_code)
        
        # Generate README
        readme_content = generate_model_readme(model_name, model_data, trainer)
        readme_path = os.path.join(app.config['MODELS_FOLDER'], f"{base_filename}_README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"Model {model_name} saved successfully")
        
        return jsonify({
            'success': True,
            'message': f'Model {model_name} saved successfully',
            'model_path': f'/api/model/download/{base_filename}.pkl',
            'prediction_code_path': f'/api/model/download/{base_filename}_predict.py',
            'readme_path': f'/api/model/download/{base_filename}_README.md',
            'files': {
                'model': f"{base_filename}.pkl",
                'code': f"{base_filename}_predict.py",
                'readme': f"{base_filename}_README.md"
            }
        })
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/model/download/<filename>', methods=['GET'])
def download_model_file(filename):
    """Download model files."""
    try:
        filepath = os.path.join(app.config['MODELS_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=filename)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_prediction_code(model_name, base_filename, trainer):
    """Generate Python prediction code for the saved model."""
    
    task_type = session.get('task_type', 'classification')
    target_col = session.get('target_column', 'target')
    
    # Get preprocessor to extract feature information
    preprocessor = get_preprocessor()
    feature_cols = list(preprocessor.X.columns) if preprocessor and preprocessor.X is not None else []
    
    # Create feature list string
    features_str = str(feature_cols)
    
    code = f'''"""
Prediction Script for {model_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Task Type: {task_type}

USAGE:
    python {base_filename}_predict.py input.csv

CSV FORMAT:
    - First row: Feature names (column headers)
    - Second row: Feature values
    - Example: age,sex,cp,trestbps,chol
              63,1,3,145,233
"""

import sys
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Expected features (in order after preprocessing)
EXPECTED_FEATURES = {features_str}

# Load the trained model
model = joblib.load('{base_filename}.pkl')

print("=" * 80)
print("Model loaded successfully!")
print("Model: {model_name}")
print("Task Type: {task_type}")
print("=" * 80)
print()


def predict_from_csv(csv_path):
    """
    Make prediction from CSV file.
    
    CSV should have:
    - Row 1: Feature names (headers)
    - Row 2: Feature values
    
    The script will automatically map and preprocess features.
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        print(f"Loaded CSV with {{len(df)}} row(s) and {{len(df.columns)}} column(s)")
        print(f"Columns: {{list(df.columns)}}")
        print()
        
        # Create a dataframe with expected features
        processed_df = pd.DataFrame(columns=EXPECTED_FEATURES)
        
        # Map available features
        for col in df.columns:
            if col in EXPECTED_FEATURES:
                processed_df[col] = df[col]
        
        # Fill missing features with 0
        for col in EXPECTED_FEATURES:
            if col not in processed_df.columns or processed_df[col].isna().all():
                processed_df[col] = 0
        
        # Ensure correct order
        processed_df = processed_df[EXPECTED_FEATURES]
        
        print("=" * 80)
        print("MAKING PREDICTION")
        print("=" * 80)
        print()
        
        # Make prediction
        prediction = model.predict(processed_df)
        
        print(f"Prediction: {{prediction[0]}}")
        print()
        
        # Get probabilities if classification
        if hasattr(model, 'predict_proba') and '{task_type}' == 'classification':
            probabilities = model.predict_proba(processed_df)
            print("Class Probabilities:")
            for i, prob in enumerate(probabilities[0]):
                print(f"  Class {{i}}: {{prob:.4f}} ({{prob*100:.2f}}%)")
            print()
            print(f"Confidence: {{max(probabilities[0]):.4f}} ({{max(probabilities[0])*100:.2f}}%)")
        
        print()
        print("=" * 80)
        print("PREDICTION COMPLETE")
        print("=" * 80)
        
        return prediction[0]
        
    except Exception as e:
        print(f"ERROR: {{str(e)}}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 80)
        print("ERROR: No CSV file provided!")
        print("=" * 80)
        print()
        print("USAGE:")
        print(f"    python {{sys.argv[0]}} input.csv")
        print()
        print("CSV FORMAT:")
        print("    Row 1: Feature names (column headers)")
        print("    Row 2: Feature values")
        print()
        print("EXAMPLE CSV (input.csv):")
        print("    age,sex,cp,trestbps,chol")
        print("    63,1,3,145,233")
        print()
        print("=" * 80)
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    print()
    print("=" * 80)
    print(f"PREDICTION SCRIPT: {{'{model_name}'}}")
    print("=" * 80)
    print(f"Input file: {{csv_file}}")
    print()
    
    result = predict_from_csv(csv_file)
    
    if result is not None:
        print()
        print("[SUCCESS] Prediction successful!")
    else:
        print()
        print("[ERROR] Prediction failed!")
        sys.exit(1)
'''
    
    return code


def generate_model_readme(model_name, model_data, trainer):
    """Generate README for the saved model."""
    
    task_type = session.get('task_type', 'classification')
    target_col = session.get('target_column', 'target')
    
    # Get preprocessor to extract feature information
    preprocessor = get_preprocessor()
    feature_cols = list(preprocessor.X.columns) if preprocessor and preprocessor.X is not None else []
    
    readme = f'''# {model_name} - Model Documentation

## Model Information

- **Model Name:** {model_name}
- **Task Type:** {task_type}
- **Target Variable:** {target_col}
- **Number of Features:** {len(feature_cols)}
- **Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics

'''
    
    # Add metrics
    if 'score' in model_data:
        readme += f"- **Score:** {model_data['score']:.4f}\n"
    if 'training_time' in model_data:
        readme += f"- **Training Time:** {model_data['training_time']:.2f} seconds\n"
    
    readme += f'''

## Features Used

The model was trained on the following features:

'''
    
    for i, col in enumerate(feature_cols[:20], 1):  # Limit to first 20
        readme += f"{i}. {col}\n"
    
    if len(feature_cols) > 20:
        readme += f"\n... and {len(feature_cols) - 20} more features\n"
    
    readme += f'''

## How to Use

### Quick Start - Using CSV File

1. **Create a CSV file** with your input data:
   - Row 1: Feature names (column headers)
   - Row 2: Feature values

Example `input.csv`:
```
age,sex,cp,trestbps,chol
63,1,3,145,233
```

2. **Run the prediction script**:
```bash
python [model_file]_predict.py input.csv
```

The script will:
- Load the model automatically
- Map your features to the expected format
- Handle missing features (fills with 0)
- Make prediction
- Show probabilities (for classification)
- Display confidence score

### Advanced Usage

#### Load the Model Manually

```python
import joblib
model = joblib.load('{model_name.replace(" ", "_")}_[timestamp].pkl')
```

#### Make Predictions Programmatically

```python
import pandas as pd

# Prepare your data (must match training features)
data = pd.DataFrame({{
    # Add your feature values here
}})

# Make prediction
prediction = model.predict(data)
print(prediction)
```

#### Get Prediction Probabilities (Classification only)

```python
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(data)
    confidence = max(probabilities[0])
    print(f"Prediction: {{prediction[0]}}")
    print(f"Confidence: {{confidence:.2%}}")
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib

Install requirements:

```bash
pip install pandas numpy scikit-learn joblib
```

## Important Notes

1. **Data Format:** Ensure your input data has the same features and format as the training data
2. **Preprocessing:** Apply the same preprocessing steps used during training
3. **Missing Values:** Handle missing values before making predictions
4. **Categorical Variables:** Encode categorical variables if they were encoded during training

## Model Details

- **Algorithm:** {model_name}
- **Task:** {task_type.capitalize()}
- **Input:** {len(feature_cols)} features
- **Output:** Prediction for '{target_col}'

## Support

For questions or issues, refer to the ML Platform documentation.

---

*Generated by Aymen's ML Platform*
*© 2025 Aymen's Labs*
'''
    
    return readme


# ============================================================================
# API ENDPOINTS - SESSION MANAGEMENT
# ============================================================================

@app.route('/api/session/status', methods=['GET'])
def get_session_status():
    """Get current session status."""
    return jsonify({
        'success': True,
        'session_id': get_session_id(),
        'status': {
            'data_loaded': session.get('data_loaded', False),
            'preprocessing_done': session.get('preprocessing_done', False),
            'training_done': session.get('training_done', False),
            'evaluation_done': session.get('evaluation_done', False),
            'filename': session.get('filename'),
            'task_type': session.get('task_type'),
            'target_column': session.get('target_column'),
            'n_features': len(session.get('feature_columns', []))
        }
    })


@app.route('/api/session/reset', methods=['POST'])
def reset_session():
    """Reset current session."""
    session_id = get_session_id()
    
    # Clear session data
    session.clear()
    
    # Remove from global objects
    if session_id in data_loaders:
        del data_loaders[session_id]
    if session_id in preprocessors:
        del preprocessors[session_id]
    if session_id in visualizers:
        del visualizers[session_id]
    if session_id in trainers:
        del trainers[session_id]
    if session_id in evaluators:
        del evaluators[session_id]
    
    return jsonify({
        'success': True,
        'message': 'Session reset successfully'
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('index.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Aymen's ML Platform - Automated Machine Learning Platform")
    print("=" * 80)
    print("\n(c) 2025 Aymen's Labs - All Rights Reserved")
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5001")
    print("\nServer Status: Running")
    print("Press CTRL+C to stop the server")
    print("=" * 80)
    print()
    
    app.run(debug=True, host='127.0.0.1', port=5001, use_reloader=False)
