"""
Main script for equity performance gap analysis.
"""

import sys
import os
sys.path.append('03_src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_all_data
from feature_engineering import create_all_features
from utils import prepare_data_splits, train_all_models, save_all_results, display_linear_comparison
from visualization import plot_correlation_matrix, plot_pairplot, plot_actual_vs_predicted_comparison


def main():
    
    print("="*80)
    print("EQUITY PERFORMANCE GAP ANALYSIS")
    print("="*80)
    
    print("\nLoading data...")
    df_all = load_all_data()
    print(f"Data loaded: {df_all.shape[0]} observations")
    
    print("\nCreating features...")
    df_features = create_all_features(df_all)
    print(f"Features created: {df_features.shape}")
    
    df_features.to_csv('02_data_processed/model_data.csv')
    print("Model data saved to 02_data_processed/model_data.csv")
    
    print("\nGenerating exploratory visualizations...")
    plot_correlation_matrix(df_features, save_path='05_figures/correlation_matrix.png')
    print("Saved correlation matrix to 05_figures/correlation_matrix.png")
    
    plot_pairplot(df_features, save_path='05_figures/pairplot.png')
    print("Saved pairplot to 05_figures/pairplot.png")
    
    print("\nPreparing train/test split...")
    data = prepare_data_splits(df_features)
    print(f"Train: {data['train_size']} obs, Test: {data['test_size']} obs")
    
    print("\nTraining models...")
    models, results, predictions = train_all_models(data)
    print("Models trained successfully")
    
    print("\nModel Comparison Results")
    print("="*50)
    
    # Creating the list of dictionaries for linear models comparison
    linear_models_info = [
        {'name': 'M1: Simple', 'n_vars': 4, 'r2_test': results['M1']['R²'], 'r2_train': 0, 'mse_test': results['M1']['MSE'], 'n_train': len(data['y_train']), 'n_test': len(data['y_test'])},
        {'name': 'M2a: Polynomial', 'n_vars': 6, 'r2_test': results['M2a']['R²'], 'r2_train': 0, 'mse_test': results['M2a']['MSE'], 'n_train': len(data['y_train']), 'n_test': len(data['y_test'])},
        {'name': 'M2b: Lags', 'n_vars': 8, 'r2_test': results['M2b']['R²'], 'r2_train': 0, 'mse_test': results['M2b']['MSE'], 'n_train': len(data['y_train_lags']), 'n_test': len(data['y_test_lags'])},
        {'name': 'M3: Lasso + Lags', 'n_vars': results['M3']['n_vars'], 'r2_test': results['M3']['R²'], 'r2_train': 0, 'mse_test': results['M3']['MSE'], 'n_train': len(data['y_train_lags']), 'n_test': len(data['y_test_lags'])}
    ]
    
    from sklearn import metrics
    linear_models_info[0]['r2_train'] = metrics.r2_score(data['y_train'], models['M1'].predict(data['X_train']))
    linear_models_info[1]['r2_train'] = metrics.r2_score(data['y_train'], models['M2a'].predict(data['X_train_poly']))
    linear_models_info[2]['r2_train'] = metrics.r2_score(data['y_train_lags'], models['M2b'].predict(data['X_train_lags']))
    linear_models_info[3]['r2_train'] = metrics.r2_score(data['y_train_lags'], models['M3'].predict(data['X_train_lags']))
    
    display_linear_comparison(linear_models_info)
    
    print("\n" + "="*50)
    print("ALL MODELS COMPARISON")
    print("="*50)
    
    comparison = pd.DataFrame({
        'Model': ['M1: Simple', 'M2a: Polynomial', 'M2b: Lags', 'M3: Lasso+Lags', 'M4: Random Forest', 'M5: Gradient Boosting'],
        'Type': ['Linear', 'Linear', 'Linear', 'Linear (Regularized)', 'Non-Linear', 'Non-Linear'],
        'Variables': [results['M1']['n_vars'], results['M2a']['n_vars'], results['M2b']['n_vars'], results['M3']['n_vars'], results['M4']['n_vars'], results['M5']['n_vars']],
        'R2 Test': [results['M1']['R²'], results['M2a']['R²'], results['M2b']['R²'], results['M3']['R²'], results['M4']['R²'], results['M5']['R²']],
        'RMSE': [results['M1']['RMSE'], results['M2a']['RMSE'], results['M2b']['RMSE'], results['M3']['RMSE'], results['M4']['RMSE'], results['M5']['RMSE']],
        'MAE': [results['M1']['MAE'], results['M2a']['MAE'], results['M2b']['MAE'], results['M3']['MAE'], results['M4']['MAE'], results['M5']['MAE']]
    })
    
    print("\nALL MODELS COMPARISON:")
    print(comparison.to_string(index=False))
    
    best_idx = comparison['R2 Test'].idxmax()
    best_model = comparison.loc[best_idx, 'Model']
    best_r2 = comparison.loc[best_idx, 'R2 Test']
    
    print(f"\nBEST MODEL: {best_model} (R2 = {best_r2:.4f})")
    print("="*60)
    
    print("\nVisualizing residuals for best model...")
    if best_model == 'M1: Simple':
        model_key = 'M1'
        y_test_best = data['y_test']
    elif best_model == 'M2a: Polynomial':
        model_key = 'M2a'
        y_test_best = data['y_test']
    elif best_model == 'M2b: Lags':
        model_key = 'M2b'
        y_test_best = data['y_test_lags']
    elif best_model == 'M3: Lasso+Lags':
        model_key = 'M3'
        y_test_best = data['y_test_lags']
    elif best_model == 'M4: Random Forest':
        model_key = 'M4'
        y_test_best = data['y_test_lags']
    else:
        model_key = 'M5'
        y_test_best = data['y_test_lags']
    
    y_pred_best = predictions[model_key]
    
    # Figure : Actual vs Predicted for 3 models
    print("\nGenerating actual vs predicted plots...")
    test_dates = data['df_lags'].loc['2024-01-31':].index
    
    models_to_plot = {
        'M2b: Lags (Best R2)': predictions['M2b'],
        'M3: Lasso (Best AIC/BIC)': predictions['M3'],
        'M1: Simple': predictions['M1']
    }
    
    plot_actual_vs_predicted_comparison(models_to_plot, data['y_test_lags'], test_dates, 
                                       save_path='05_figures/actual_vs_predicted.png')
    print("Saved figure to 05_figures/actual_vs_predicted.png")
    
    # Overfitting analysis
    print("\n" + "="*50)
    print("OVERFITTING ANALYSIS")
    print("="*50)
    print("\nTrain vs Test R² Comparison:")
    print(f"{'Model':<25} {'R² Train':>10} {'R² Test':>10} {'Gap':>10} {'Status':<20}")
    print("-"*80)
    
    from sklearn import metrics
    
    r2_train_m1 = metrics.r2_score(data['y_train'], models['M1'].predict(data['X_train']))
    r2_train_m2a = metrics.r2_score(data['y_train'], models['M2a'].predict(data['X_train_poly']))
    r2_train_m2b = metrics.r2_score(data['y_train_lags'], models['M2b'].predict(data['X_train_lags']))
    r2_train_m3 = metrics.r2_score(data['y_train_lags'], models['M3'].predict(data['X_train_lags']))
    r2_train_m4 = metrics.r2_score(data['y_train_lags'], models['M4'].predict(data['X_train_sel']))
    r2_train_m5 = metrics.r2_score(data['y_train_lags'], models['M5'].predict(data['X_train_sel']))
    
    r2_tests = {
        'M1': comparison.loc[comparison['Model'] == 'M1: Simple', 'R2 Test'].values[0],
        'M2a': comparison.loc[comparison['Model'] == 'M2a: Polynomial', 'R2 Test'].values[0],
        'M2b': comparison.loc[comparison['Model'] == 'M2b: Lags', 'R2 Test'].values[0],
        'M3': comparison.loc[comparison['Model'] == 'M3: Lasso+Lags', 'R2 Test'].values[0],
        'M4': comparison.loc[comparison['Model'] == 'M4: Random Forest', 'R2 Test'].values[0],
        'M5': comparison.loc[comparison['Model'] == 'M5: Gradient Boosting', 'R2 Test'].values[0]
    }
    
    overfitting_data = [
        ('M1: Simple', r2_train_m1, r2_tests['M1']),
        ('M2a: Polynomial', r2_train_m2a, r2_tests['M2a']),
        ('M2b: Lags', r2_train_m2b, r2_tests['M2b']),
        ('M3: Lasso+Lags', r2_train_m3, r2_tests['M3']),
        ('M4: Random Forest', r2_train_m4, r2_tests['M4']),
        ('M5: Gradient Boosting', r2_train_m5, r2_tests['M5'])
    ]
    
    for model_name, r2_train, r2_test in overfitting_data:
        gap = r2_train - r2_test
        if gap < 0.15:
            status = "Good balance"
        elif gap < 0.30:
            status = "Medium overfitting"
        else:
            status = "High overfitting "
        
        print(f"{model_name:<25} {r2_train:>10.4f} {r2_test:>10.4f} {gap:>10.4f} {status:<20}")
    
    print("\nNote: Gap > 0.15 indicates medium overfitting, gap > 0.30 indicates high overfitting.")
    print("="*50)
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    print("\nLasso (M3) - Selected Variables:")
    feature_names = ['X1', 'X2', 'X3', 'X4', 'X1_lag1', 'X2_lag1', 'X3_lag1', 'X4_lag1']
    lasso_coefs = models['M3'].coef_
    for i in range(len(feature_names)):
        if lasso_coefs[i] != 0:
            print(f"  {feature_names[i]}: {lasso_coefs[i]:.4f}")
    
    print("\nRandom Forest (M4) - Feature Importance:")
    rf_importance = models['M4'].feature_importances_
    rf_features = ['X1', 'X2', 'X3', 'X4', 'X3_lag1']
    df_rf_imp = pd.DataFrame({'Feature': rf_features, 'Importance': rf_importance})
    df_rf_imp = df_rf_imp.sort_values(by='Importance', ascending=False)
    for idx, row in df_rf_imp.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    print("\nGradient Boosting (M5) - Feature Importance:")
    gb_importance = models['M5'].feature_importances_
    df_gb_imp = pd.DataFrame({'Feature': rf_features, 'Importance': gb_importance})
    df_gb_imp = df_gb_imp.sort_values(by='Importance', ascending=False)
    for idx, row in df_gb_imp.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    print("="*50)
    
    lasso_data = {'Feature': feature_names, 'Lasso_Coef': lasso_coefs}
    df_lasso = pd.DataFrame(lasso_data)
    df_lasso['Selected'] = df_lasso['Lasso_Coef'].apply(lambda x: 'Yes' if x != 0 else 'No')
    
    df_rf = pd.DataFrame({'Feature': rf_features, 'RF_Importance': rf_importance})
    df_gb = pd.DataFrame({'Feature': rf_features, 'GB_Importance': gb_importance})
    
    feature_importance = df_lasso.merge(df_rf, on='Feature', how='left').merge(df_gb, on='Feature', how='left')
    feature_importance.to_csv('06_results/feature_importance.csv', index=False)
    
    print("\nSaving results...")
    linear_models_info_full, comparison_full, best_model_name, best_r2_val = save_all_results(models, results, predictions, data)
    
    print("Saved linear comparison to 06_results/linear_comparison.csv")
    print("Saved model comparison to 06_results/model_comparison.csv")
    print("Saved best model info to 06_results/best_model.txt")
    print("Saved feature importance to 06_results/feature_importance.csv")
    print("Saved visualization to 05_figures/model_comparison.png")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*50)


if __name__ == "__main__":
    main()
