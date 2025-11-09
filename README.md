# Aymen's ML Platform - Automated Machine Learning Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**A professional-grade, end-to-end machine learning web application that automates the entire ML workflow from data loading to model deployment and reporting.**

**© 2025 Aymen's Labs - All Rights Reserved**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Documentation](#documentation)

</div>

---

## 🎯 Overview

Aymen's ML Platform is a comprehensive, automated machine learning platform designed for data scientists, ML engineers, and analysts who want to streamline their workflow. Built with Flask and modern ML libraries, it provides a complete GUI-based solution for:

- **Automated Data Analysis** - Smart detection of features, targets, and data types
- **Intelligent Preprocessing** - Automatic handling of missing values, encoding, and scaling
- **Interactive Visualizations** - Beautiful charts powered by Plotly
- **Multi-Level Model Training** - From basic to advanced algorithms with hyperparameter tuning
- **Comprehensive Evaluation** - Detailed metrics and comparison charts
- **Professional Reporting** - Generate publication-ready DOCX reports

## ✨ Features

### 🔍 Smart Data Detection
- Automatic detection of target variables with justifications
- Intelligent feature type classification (numerical/categorical)
- Metadata column identification (IDs, timestamps, etc.)
- Comprehensive data quality analysis

### 🔧 Advanced Preprocessing
- **Missing Value Handling**: Mean, median, mode imputation or removal
- **Encoding**: Smart auto-encoding, one-hot, or label encoding
- **Scaling**: StandardScaler, MinMaxScaler, or no scaling
- **Outlier Detection**: IQR-based outlier removal
- Full transparency with detailed transformation logs

### 📊 Interactive Visualizations
- **Missing value heatmaps** - Identify data quality issues
- **Feature distribution plots** - Understand data distributions
- **Correlation matrices** - Discover feature relationships
- **Categorical feature pie charts** - Analyze categorical data
- **Target variable analysis** - Examine target distributions
- **Outlier detection box plots** - Identify anomalies

**Chart Export Features:**
- All charts saved as high-quality PNG files (1200x600px)
- Stored in dedicated `charts/` folder
- Automatically embedded in DOCX reports
- White backgrounds for professional appearance
- Reusable across multiple report generations

### 🧠 Multi-Level Model Training

**Basic Models:**
- Logistic Regression
- Decision Tree
- Linear Regression

**Intermediate Models:**
- Random Forest
- Support Vector Machines (SVM)
- Ridge/Lasso Regression

**Advanced Models:**
- XGBoost
- Gradient Boosting
- LightGBM

**Features:**
- Automatic hyperparameter tuning with RandomizedSearchCV
- Cross-validation for robust performance estimation
- Parallel training with progress tracking
- Best model auto-selection and saving

### 📈 Comprehensive Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression Metrics**: MAE, MSE, RMSE, R², MAPE
- Confusion matrices and classification reports
- Model comparison visualizations
- Performance ranking tables

### 📄 Professional Report Generation
- **Automated DOCX report creation** - One-click report generation
- **Dataset Summary** - Complete data statistics and information
- **Preprocessing Details** - All transformations documented
- **Data Visualizations** - 6 high-quality charts embedded (NEW!)
  - Missing values analysis
  - Feature distributions
  - Correlation matrix
  - Categorical features
  - Target distribution
  - Outliers detection
- **Model Comparison** - Performance metrics for all models
- **Best Model Details** - Hyperparameters and training info
- **Conclusions & Recommendations** - Actionable insights
- **Professional Formatting** - Publication-ready quality
- Ready for distribution to stakeholders

## 🚀 Quick Start

### ⚠️ IMPORTANT: Run This First!
```bash
python test_and_run.py
```
This script will:
- ✅ Check Python version (3.8+ required)
- ✅ Verify all dependencies (including **kaleido**)
- ✅ Auto-install missing packages
- ✅ Test all modules
- ✅ Launch the application

### Manual Installation (Alternative)
```bash
# Install all dependencies
pip install -r requirements.txt

# Start the application
python app.py
```

### Access the Application
Navigate to: **http://localhost:5001**

**That's it!** 🎉

---

## 🎮 Two Ways to Run

### Option 1: Automated (Recommended) - `test_and_run.py`
```bash
python test_and_run.py
```
**What it does:**
- ✅ Checks Python version (3.8+)
- ✅ Verifies ALL 14 dependencies
- ✅ Auto-installs missing packages
- ✅ Tests all module imports
- ✅ Verifies Flask app configuration
- ✅ Offers to launch the application
- ✅ Opens at http://localhost:5001

**Use this when:**
- First time setup
- After pulling from GitHub
- When dependencies might be missing
- To verify everything works

### Option 2: Direct Launch - `app.py`
```bash
python app.py
```
**What it does:**
- Starts Flask server immediately
- Opens at http://localhost:5001
- Assumes all dependencies installed

**Use this when:**
- You've already run `test_and_run.py` once
- All dependencies are confirmed working
- Quick restart during development

---

## ⚙️ Critical Dependencies

### 🔴 Kaleido (Required for Chart Export)

**This is CRITICAL for report generation with visualizations!**

The platform uses **Plotly** for interactive charts and **Kaleido** to export them as static images for DOCX reports.

**Required Version:** `kaleido==0.2.1`

**Why this specific version?**
- Plotly 5.18.0 is **NOT compatible** with Kaleido 1.2.0+
- Using wrong version will cause charts to **NOT appear** in reports
- Version 0.2.1 is the **stable, compatible version**

**The `test_and_run.py` script automatically checks for this!**

If you experience issues with chart generation:
```bash
pip uninstall kaleido -y
pip install kaleido==0.2.1
```

## 📖 Usage

1. **Upload Data** - Upload CSV file
2. **Preprocess** - Configure and clean data
3. **Visualize** - Explore with interactive charts
4. **Train** - Train multiple ML models
5. **Evaluate** - Compare model performance
6. **Report** - Generate DOCX report

## 🏗️ Architecture

### Project Structure
```
ml_auto_project/
│
├── app.py                      # Flask application entry point
├── test_and_run.py            # Automated test & run script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── ml_pipeline/                # Core ML pipeline modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and detection
│   ├── preprocessing.py       # Data preprocessing
│   ├── visualization.py       # Data visualization
│   ├── model_training.py      # Model training and tuning
│   ├── evaluation.py          # Model evaluation
│   ├── reporting.py           # Report generation
│   └── utils.py               # Utility functions
│
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Landing page
│   ├── data.html              # Data loading page
│   ├── preprocess.html        # Preprocessing page
│   ├── visualize.html         # Visualization page
│   ├── train.html             # Training page
│   ├── evaluate.html          # Evaluation page
│   └── documentation.html     # Report generation page
│
├── static/                     # Static assets
│   ├── styles.css             # Custom CSS (Dark theme)
│   └── scripts.js             # JavaScript utilities
│
├── uploads/                    # User uploaded datasets
├── models/                     # Saved models
├── charts/                     # Generated chart PNG files
└── reports/                    # Generated DOCX reports
```

### Technology Stack

**Backend:**
- Flask 3.0.0 - Web framework
- Scikit-learn 1.3.2 - ML algorithms
- XGBoost 2.0.3 - Gradient boosting
- LightGBM 4.1.0 - Gradient boosting
- Pandas 1.5.3 - Data manipulation
- NumPy 1.24.3 - Numerical computing

**Visualization & Reporting:**
- Plotly 5.18.0 - Interactive charts
- Kaleido 0.2.1 - Chart export (CRITICAL!)
- python-docx 1.1.0 - DOCX generation
- Matplotlib 3.7.1 - Static plots
- Seaborn 0.12.2 - Statistical visualizations

**Frontend:**
- Bootstrap 5.3 - UI framework
- Plotly.js - Interactive charts
- Font Awesome 6.4 - Icons
- Custom Dark Theme - Professional UI
- Vanilla JavaScript - Client-side logic

## 🔧 Configuration

### Application Settings

Edit `app.py` to customize:

```python
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['MODELS_FOLDER'] = 'models'
app.config['CHARTS_FOLDER'] = 'charts'  # Chart PNG storage
app.config['REPORTS_FOLDER'] = 'reports'
```

### Model Configuration

Modify `ml_pipeline/model_training.py` to add custom models:

```python
CLASSIFICATION_MODELS = {
    'basic': {
        'Your Model': {
            'model': YourModelClass,
            'params': {...}
        }
    }
}
```

## 📊 API Endpoints

### Data Management
- `POST /api/upload` - Upload dataset
- `POST /api/detect_columns` - Auto-detect column roles
- `POST /api/confirm_columns` - Confirm column selections

### Preprocessing
- `POST /api/preprocess` - Execute preprocessing pipeline

### Visualization
- `GET /api/visualize/all` - Generate all plots
- `GET /api/visualize/<plot_type>` - Generate specific plot

### Training
- `GET /api/models/available` - Get available models
- `POST /api/train` - Train selected models

### Evaluation
- `POST /api/evaluate` - Evaluate all models
- `POST /api/evaluate/plot/<plot_type>` - Get evaluation plots

### Reporting
- `POST /api/report/generate` - Generate DOCX report
- `GET /api/report/download/<filename>` - Download report

### Session
- `GET /api/session/status` - Get session status
- `POST /api/session/reset` - Reset session

## 🎓 Examples

### Example 1: Classification Task
```python
# Upload iris.csv or any classification dataset
# System will automatically:
# 1. Detect target column (e.g., 'species')
# 2. Identify numerical features (sepal_length, sepal_width, etc.)
# 3. Preprocess data (scaling, encoding)
# 4. Train multiple classifiers
# 5. Compare accuracy, precision, recall, F1-score
# 6. Generate comprehensive report
```

### Example 2: Regression Task
```python
# Upload housing.csv or any regression dataset
# System will automatically:
# 1. Detect target column (e.g., 'price')
# 2. Handle missing values and outliers
# 3. Scale features appropriately
# 4. Train regression models
# 5. Compare R², RMSE, MAE
# 6. Generate prediction vs actual plots
```

## 🛠️ Troubleshooting

### Common Issues

**🔴 CRITICAL: Charts not appearing in DOCX reports**
```bash
# This is the #1 issue! It's caused by wrong Kaleido version

# Solution:
pip uninstall kaleido -y
pip install kaleido==0.2.1

# Verify installation:
python -c "import plotly.graph_objects as go; fig = go.Figure(data=[go.Bar(x=[1,2,3], y=[4,5,6])]); fig.write_image('test.png'); print('SUCCESS!')"

# If you see "SUCCESS!" then it's working!
```

**Why this happens:**
- Kaleido 1.2.0+ is NOT compatible with Plotly 5.18.0
- When incompatible, `fig.write_image()` fails silently
- Charts won't be saved or embedded in reports
- **Always use kaleido==0.2.1 with this platform**

**Issue: Port 5001 already in use**
```bash
# Solution: Change port in app.py (line 1286)
# Change from:
app.run(debug=True, host='127.0.0.1', port=5001, use_reloader=False)

# To (use a different port):
app.run(debug=True, host='127.0.0.1', port=5002, use_reloader=False)
```

**Issue: Charts folder missing**
```bash
# Solution: The folder is created automatically on startup
# If it doesn't exist, create it manually:
mkdir charts

# Or just restart the application:
python app.py
```

**Issue: Memory error with large datasets**
```bash
# Solution: Increase system memory or sample data
# The system automatically handles datasets up to 100MB
```

**Issue: Model training takes too long**
```bash
# Solution: Reduce hyperparameter search iterations
# In training page, set "Hyperparameter Search Iterations" to 10
```

**Issue: Missing dependencies**
```bash
# Solution: Use test_and_run.py - it auto-installs everything!
python test_and_run.py

# Or manually:
pip install --upgrade -r requirements.txt
```

## 🔒 Security Considerations

- **File Upload**: Only CSV files are accepted
- **File Size**: Limited to 100MB by default
- **Session Management**: Uses Flask sessions (change secret_key in production)
- **Input Validation**: All user inputs are validated
- **Path Traversal**: Secure filename handling with werkzeug

**For Production Deployment:**
1. Change `app.secret_key` to a secure random value
2. Set `debug=False` in `app.run()`
3. Use a production WSGI server (Gunicorn, uWSGI)
4. Implement proper authentication and authorization
5. Use HTTPS with SSL certificates
6. Set up proper logging and monitoring

## 📝 Best Practices

1. **Data Preparation**: Clean your CSV before upload (remove special characters from column names)
2. **Feature Selection**: Review auto-detected features and adjust if needed
3. **Preprocessing**: Start with default settings, then customize based on results
4. **Model Selection**: Begin with basic models, add advanced if needed
5. **Evaluation**: Always check multiple metrics, not just accuracy
6. **Reporting**: Generate reports after each successful run for documentation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional ML algorithms (Neural Networks, Ensemble methods)
- More visualization types (3D plots, interactive dashboards)
- Database integration for result persistence
- User authentication and multi-user support
- API for programmatic access
- Docker containerization
- Cloud deployment guides

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

**Aymen's ML Platform** - Built with ❤️ by Aymen's Labs

Technologies:
- Flask - Web framework
- Scikit-learn - ML library
- XGBoost & LightGBM - Gradient boosting
- Plotly - Interactive visualizations
- Bootstrap - UI framework

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**© 2025 Aymen's Labs - All Rights Reserved**

---

<div align="center">

**Aymen's ML Platform - Built with ❤️ by Aymen's Labs**

**© 2025 Aymen's Labs - All Rights Reserved**

⭐ Star this project if you find it useful!

</div>
