"""
Aymen's ML Platform - Test & Run Script
========================================
Comprehensive testing and verification script

© 2025 Aymen's Labs - All Rights Reserved
"""

import sys
import os
import subprocess
import importlib.util

def print_header():
    """Print branded header."""
    print("\n" + "=" * 80)
    print("🚀 Aymen's ML Platform - Test & Run Script")
    print("=" * 80)
    print("© 2025 Aymen's Labs - All Rights Reserved")
    print("=" * 80 + "\n")

def print_section(title):
    """Print section header."""
    print(f"\n{'─' * 80}")
    print(f"📋 {title}")
    print(f"{'─' * 80}")

def check_python_version():
    """Check Python version compatibility."""
    print_section("Step 1: Checking Python Version")
    
    version = sys.version_info
    print(f"   Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ FAILED: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print("   ✅ PASSED: Python version is compatible")
    return True

def check_required_packages():
    """Check if all required packages are installed."""
    print_section("Step 2: Checking Required Packages")
    
    required_packages = {
        'flask': 'Flask',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'plotly': 'Plotly',
        'kaleido': 'Kaleido',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'docx': 'python-docx',
        'joblib': 'Joblib',
        'scipy': 'SciPy',
        'PIL': 'Pillow'
    }
    
    missing_packages = []
    installed_packages = []
    
    for package, display_name in required_packages.items():
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                print(f"   ✅ {display_name:20s} - Installed")
                installed_packages.append(display_name)
            else:
                print(f"   ❌ {display_name:20s} - NOT FOUND")
                missing_packages.append(display_name)
        except (ImportError, ModuleNotFoundError):
            print(f"   ❌ {display_name:20s} - NOT FOUND")
            missing_packages.append(display_name)
    
    print(f"\n   Summary: {len(installed_packages)}/{len(required_packages)} packages installed")
    
    if missing_packages:
        print(f"\n   ⚠️  Missing packages: {', '.join(missing_packages)}")
        print("\n   To install missing packages, run:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n   ✅ PASSED: All required packages are installed")
    return True

def check_directory_structure():
    """Check if all required directories exist."""
    print_section("Step 3: Checking Directory Structure")
    
    required_dirs = [
        'ml_pipeline',
        'templates',
        'static',
        'uploads',
        'models',
        'reports'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"   ✅ {directory:20s} - Exists")
        else:
            print(f"   ❌ {directory:20s} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n   ⚠️  Some directories are missing. Creating them...")
        for directory in required_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"   ✅ Created: {directory}/")
    
    print("\n   ✅ PASSED: All directories exist")
    return True

def check_required_files():
    """Check if all required files exist."""
    print_section("Step 4: Checking Required Files")
    
    required_files = {
        'app.py': 'Main Flask application',
        'requirements.txt': 'Dependencies file',
        'README.md': 'Documentation',
        'ml_pipeline/__init__.py': 'ML Pipeline package',
        'ml_pipeline/data_loader.py': 'Data loader module',
        'ml_pipeline/preprocessing.py': 'Preprocessing module',
        'ml_pipeline/visualization.py': 'Visualization module',
        'ml_pipeline/model_training.py': 'Training module',
        'ml_pipeline/evaluation.py': 'Evaluation module',
        'ml_pipeline/reporting.py': 'Reporting module',
        'ml_pipeline/utils.py': 'Utilities module',
        'templates/index.html': 'Home page template',
        'templates/data.html': 'Data page template',
        'static/styles.css': 'CSS stylesheet',
        'static/scripts.js': 'JavaScript file'
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"   ✅ {description:30s} - Exists")
        else:
            print(f"   ❌ {description:30s} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n   ❌ FAILED: Some required files are missing")
        return False
    
    print("\n   ✅ PASSED: All required files exist")
    return True

def test_imports():
    """Test if all modules can be imported."""
    print_section("Step 5: Testing Module Imports")
    
    modules_to_test = [
        ('ml_pipeline', 'ML Pipeline Package'),
        ('ml_pipeline.data_loader', 'Data Loader'),
        ('ml_pipeline.preprocessing', 'Preprocessing'),
        ('ml_pipeline.visualization', 'Visualization'),
        ('ml_pipeline.model_training', 'Model Training'),
        ('ml_pipeline.evaluation', 'Evaluation'),
        ('ml_pipeline.reporting', 'Reporting'),
        ('ml_pipeline.utils', 'Utilities')
    ]
    
    all_imported = True
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"   ✅ {description:30s} - Import successful")
        except Exception as e:
            print(f"   ❌ {description:30s} - Import failed: {str(e)}")
            all_imported = False
    
    if not all_imported:
        print("\n   ❌ FAILED: Some modules could not be imported")
        return False
    
    print("\n   ✅ PASSED: All modules imported successfully")
    return True

def check_flask_app():
    """Check if Flask app can be initialized."""
    print_section("Step 6: Testing Flask Application")
    
    try:
        # Try to import the app
        sys.path.insert(0, os.getcwd())
        from app import app as flask_app
        
        print("   ✅ Flask app imported successfully")
        print(f"   ✅ App name: {flask_app.name}")
        print(f"   ✅ Secret key configured: {'Yes' if flask_app.secret_key else 'No'}")
        
        # Check routes
        routes = [rule.rule for rule in flask_app.url_map.iter_rules()]
        print(f"   ✅ Number of routes: {len(routes)}")
        
        print("\n   ✅ PASSED: Flask application is properly configured")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: Could not initialize Flask app")
        print(f"   Error: {str(e)}")
        return False

def print_summary(results):
    """Print test summary."""
    print_section("Test Summary")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    print(f"\n   Total Tests: {total_tests}")
    print(f"   ✅ Passed: {passed_tests}")
    print(f"   ❌ Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("\n   🎉 ALL TESTS PASSED! 🎉")
        print("\n   Your Aymen's ML Platform is ready to run!")
        return True
    else:
        print("\n   ⚠️  SOME TESTS FAILED")
        print("\n   Please fix the issues above before running the application.")
        return False

def offer_to_run():
    """Offer to start the application."""
    print("\n" + "=" * 80)
    print("🚀 Ready to Launch Aymen's ML Platform")
    print("=" * 80)
    
    print("\nWould you like to start the application now? (y/n): ", end='')
    response = input().strip().lower()
    
    if response == 'y':
        print("\n" + "=" * 80)
        print("Starting Aymen's ML Platform...")
        print("=" * 80)
        print("\n📍 The application will be available at: http://localhost:5001")
        print("🛑 Press CTRL+C to stop the server\n")
        print("=" * 80 + "\n")
        
        try:
            # Run the Flask app
            subprocess.run([sys.executable, 'app.py'])
        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("🛑 Server stopped by user")
            print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("To start the application manually, run:")
        print("   python app.py")
        print("\nThen open your browser and navigate to:")
        print("   http://localhost:5001")
        print("=" * 80)

def main():
    """Main test function."""
    print_header()
    
    # Run all tests
    results = {}
    
    # Test 1: Python version
    results['Python Version'] = check_python_version()
    
    # Test 2: Required packages
    packages_ok = check_required_packages()
    results['Required Packages'] = packages_ok
    
    # If packages are missing, auto-install
    if not packages_ok:
        print("\n" + "=" * 80)
        print("📦 Installing Missing Packages")
        print("=" * 80)
        print("\n🔧 Installing automatically...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("\n✅ Packages installed! Re-checking...")
            results['Required Packages'] = check_required_packages()
        except subprocess.CalledProcessError:
            print("\n❌ Auto-install failed. Please run manually:")
            print("   pip install -r requirements.txt")
    
    # Test 3: Directory structure
    results['Directory Structure'] = check_directory_structure()
    
    # Test 4: Required files
    results['Required Files'] = check_required_files()
    
    # Test 5: Module imports (only if packages are installed)
    if results['Required Packages']:
        results['Module Imports'] = test_imports()
    else:
        results['Module Imports'] = False
        print_section("Step 5: Testing Module Imports")
        print("   ⏭️  SKIPPED: Install packages first")
    
    # If module imports failed, auto-fix
    if not results.get('Module Imports', False) and results['Required Packages']:
        print("\n" + "=" * 80)
        print("⚠️  Import Error - Version Mismatch Detected")
        print("=" * 80)
        print("\n🔧 Fixing automatically...")
        
        try:
            # Uninstall pandas
            print("\nStep 1: Removing incompatible pandas...")
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "pandas", "-y"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Reinstall from requirements
            print("Step 2: Installing compatible versions...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            print("\n✅ Fixed! Re-testing imports...")
            results['Module Imports'] = test_imports()
            
        except subprocess.CalledProcessError:
            print("\n❌ Auto-fix failed. Please run manually:")
            print("   pip uninstall pandas -y")
            print("   pip install -r requirements.txt")
    
    # Test 6: Flask app (only if modules can be imported)
    if results.get('Module Imports', False):
        results['Flask Application'] = check_flask_app()
    else:
        results['Flask Application'] = False
        print_section("Step 6: Testing Flask Application")
        print("   ⏭️  SKIPPED: Fix module imports first")
    
    # Print summary
    all_passed = print_summary(results)
    
    # Offer to run if all tests passed
    if all_passed:
        offer_to_run()
    
    print("\n© 2025 Aymen's Labs - All Rights Reserved\n")

if __name__ == "__main__":
    main()
