// Aymen's ML Platform - JavaScript Utilities
// © 2025 Aymen's Labs

// ML Auto Project - JavaScript Utilities

// Global utilities
const MLAutoProject = {
    // API base URL
    apiBase: '/api',
    
    // Show loading spinner
    showLoading: function(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'block';
        }
    },
    
    // Hide loading spinner
    hideLoading: function(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'none';
        }
    },
    
    // Show alert message
    showAlert: function(message, type = 'info', containerId = 'alertContainer') {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible fade show`;
        alert.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        container.appendChild(alert);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alert.remove();
        }, 5000);
    },
    
    // Format number
    formatNumber: function(num, decimals = 4) {
        if (typeof num !== 'number') return num;
        return num.toFixed(decimals);
    },
    
    // Format time
    formatTime: function(seconds) {
        if (seconds < 60) {
            return `${seconds.toFixed(2)}s`;
        } else if (seconds < 3600) {
            return `${(seconds / 60).toFixed(2)}m`;
        } else {
            return `${(seconds / 3600).toFixed(2)}h`;
        }
    },
    
    // Check session status
    checkSession: async function() {
        try {
            const response = await fetch(`${this.apiBase}/session/status`);
            const data = await response.json();
            return data.success ? data.status : null;
        } catch (error) {
            console.error('Failed to check session:', error);
            return null;
        }
    },
    
    // Reset session
    resetSession: async function() {
        try {
            const response = await fetch(`${this.apiBase}/session/reset`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.success) {
                this.showAlert('Session reset successfully', 'success');
                setTimeout(() => {
                    window.location.href = '/';
                }, 1500);
            }
        } catch (error) {
            console.error('Failed to reset session:', error);
            this.showAlert('Failed to reset session', 'danger');
        }
    },
    
    // Download file
    downloadFile: function(url, filename) {
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    },
    
    // Validate form
    validateForm: function(formId) {
        const form = document.getElementById(formId);
        if (!form) return false;
        
        return form.checkValidity();
    },
    
    // Enable/disable button
    toggleButton: function(buttonId, disabled = true, text = null) {
        const button = document.getElementById(buttonId);
        if (button) {
            button.disabled = disabled;
            if (text) {
                button.innerHTML = text;
            }
        }
    },
    
    // Create progress bar
    createProgressBar: function(containerId, percentage = 0) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = `
            <div class="progress">
                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                     role="progressbar" 
                     style="width: ${percentage}%"
                     aria-valuenow="${percentage}" 
                     aria-valuemin="0" 
                     aria-valuemax="100">
                    ${percentage}%
                </div>
            </div>
        `;
    },
    
    // Update progress bar
    updateProgressBar: function(containerId, percentage) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const progressBar = container.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
            progressBar.textContent = `${percentage}%`;
        }
    },
    
    // Create table from data
    createTable: function(data, columns, tableId) {
        const table = document.getElementById(tableId);
        if (!table || !data || data.length === 0) return;
        
        // Create header
        const thead = table.querySelector('thead');
        if (thead) {
            thead.innerHTML = '<tr>' + 
                columns.map(col => `<th>${col}</th>`).join('') + 
                '</tr>';
        }
        
        // Create body
        const tbody = table.querySelector('tbody');
        if (tbody) {
            tbody.innerHTML = data.map(row => 
                '<tr>' + 
                columns.map(col => {
                    const value = row[col];
                    const displayValue = typeof value === 'number' ? 
                        this.formatNumber(value) : (value || 'N/A');
                    return `<td>${displayValue}</td>`;
                }).join('') + 
                '</tr>'
            ).join('');
        }
    },
    
    // Copy to clipboard
    copyToClipboard: function(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showAlert('Copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Failed to copy:', err);
            this.showAlert('Failed to copy to clipboard', 'danger');
        });
    },
    
    // Debounce function
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Format file size
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    },
    
    // Initialize tooltips (Bootstrap)
    initTooltips: function() {
        const tooltipTriggerList = [].slice.call(
            document.querySelectorAll('[data-bs-toggle="tooltip"]')
        );
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    },
    
    // Initialize popovers (Bootstrap)
    initPopovers: function() {
        const popoverTriggerList = [].slice.call(
            document.querySelectorAll('[data-bs-toggle="popover"]')
        );
        popoverTriggerList.map(function (popoverTriggerEl) {
            return new bootstrap.Popover(popoverTriggerEl);
        });
    }
};

// Progress Tracker Management
const ProgressTracker = {
    steps: ['/', '/data', '/preprocess', '/visualize', '/train', '/evaluate', '/documentation'],
    
    init: function() {
        this.updateProgress();
        this.addClickHandlers();
    },
    
    updateProgress: function() {
        const currentPath = window.location.pathname;
        const currentIndex = this.steps.indexOf(currentPath);
        
        // Get all step elements
        const stepElements = document.querySelectorAll('.progress-step');
        const lineElements = document.querySelectorAll('.progress-line');
        
        stepElements.forEach((step, index) => {
            const stepPage = step.getAttribute('data-page');
            const stepIndex = this.steps.indexOf(stepPage);
            
            // Remove all classes first
            step.classList.remove('active', 'completed');
            
            if (stepIndex === currentIndex) {
                // Current step - active with animation
                step.classList.add('active');
            } else if (stepIndex < currentIndex) {
                // Previous steps - completed
                step.classList.add('completed');
            }
        });
        
        // Update progress lines
        lineElements.forEach((line, index) => {
            line.classList.remove('completed');
            if (index < currentIndex) {
                line.classList.add('completed');
            }
        });
    },
    
    addClickHandlers: function() {
        const stepElements = document.querySelectorAll('.progress-step');
        
        stepElements.forEach(step => {
            step.addEventListener('click', function() {
                const targetPage = this.getAttribute('data-page');
                if (targetPage) {
                    window.location.href = targetPage;
                }
            });
        });
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap components
    MLAutoProject.initTooltips();
    MLAutoProject.initPopovers();
    
    // Initialize Progress Tracker
    ProgressTracker.init();
    
    // Add fade-in animation to main content
    const mainContent = document.querySelector('.container');
    if (mainContent) {
        mainContent.classList.add('fade-in');
    }
    
    // Check session status on certain pages
    const currentPath = window.location.pathname;
    const protectedPages = ['/preprocess', '/visualize', '/train', '/evaluate', '/documentation'];
    
    if (protectedPages.includes(currentPath)) {
        MLAutoProject.checkSession().then(status => {
            if (!status || !status.data_loaded) {
                MLAutoProject.showAlert(
                    'Please upload data first. Redirecting...', 
                    'warning'
                );
                setTimeout(() => {
                    window.location.href = '/data';
                }, 2000);
            }
        });
    }
});

// Handle navigation warnings for unsaved changes
let hasUnsavedChanges = false;

window.addEventListener('beforeunload', function(e) {
    if (hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = '';
        return '';
    }
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLAutoProject;
}
