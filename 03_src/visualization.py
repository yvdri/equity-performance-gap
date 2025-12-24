import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_train_test_comparison(models_names, r2_train, r2_test):
    """Plot train vs test R² for overfitting detection."""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(models_names))
    width = 0.35
    
    plt.bar(x - width/2, r2_train, width, label='Train R²', color='blue')
    plt.bar(x + width/2, r2_test, width, label='Test R²', color='green')
    
    plt.ylabel('R²')
    plt.title('Train vs Test R²')
    plt.xticks(x, models_names, rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()


def plot_residuals(y_true, y_pred, save_path=None):
    """Plot residuals analysis."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].scatter(y_pred, residuals, color='blue')
    axes[0].axhline(y=0, linestyle='--', color='red')
    axes[0].set_xlabel('Fitted')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted')
    axes[0].grid()
    
    axes[1].hist(residuals, bins=30, color='blue')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')
    axes[1].grid()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()


def plot_predictions(dates, y_true, y_pred, title='Predictions'):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, label='Actual', color='blue', marker='o')
    plt.plot(dates, y_pred, label='Predicted', color='orange', marker='o')
    plt.title(title)
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """Plot feature importance for tree-based models (RF, GB)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[indices], color='blue')
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()


def plot_correlation_matrix(df, save_path=None):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()


def plot_pairplot(df, save_path=None):
    """Plot pairplot of all variables."""
    pairplot = sns.pairplot(df, diag_kind='hist', plot_kws={'alpha': 0.6})
    pairplot.fig.suptitle('Pairplot of Variables', y=1.01)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()


def plot_actual_vs_predicted_comparison(models_dict, y_test, dates, save_path=None):
    """Plot actual vs predicted for multiple models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['orange', 'green', 'purple']
    
    idx = 0
    for model_name, y_pred in models_dict.items():
        ax = axes[idx]
        ax.plot(dates, y_test, label='Actual', color='blue', marker='o')
        ax.plot(dates, y_pred, label='Predicted', color=colors[idx], marker='o')
        ax.set_title(f'{model_name}')
        ax.set_ylabel('Return Differential')
        ax.legend()
        ax.grid()
        ax.tick_params(axis='x', rotation=45)
        idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
