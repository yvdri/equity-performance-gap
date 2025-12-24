import pandas as pd


def create_comparison_table(models_info):
    """
    Create comparison table for multiple models.
    
    models_info: list of dict with keys:
        - name, n_vars, r2_test, mse_test, n_train, n_test
    """
    from evaluation import calculate_adjusted_r2, calculate_aic, calculate_bic
    
    results = []
    for info in models_info:
        r2_adj = calculate_adjusted_r2(info['r2_test'], info['n_train'], info['n_vars'])
        aic = calculate_aic(info['n_test'], info['mse_test'], info['n_vars'])
        bic = calculate_bic(info['n_test'], info['mse_test'], info['n_vars'])
        
        results.append({
            'Model': info['name'],
            'Vars': info['n_vars'],
            'R² Test': info['r2_test'],
            'R² Adj': r2_adj,
            'AIC': aic,
            'BIC': bic
        })
    
    return pd.DataFrame(results)


def print_best_models(df):
    """Print best model by each criterion."""
    print(f"Best by R² Test: {df.loc[df['R² Test'].idxmax(), 'Model']}")
    print(f"Best by R² Adj: {df.loc[df['R² Adj'].idxmax(), 'Model']}")
    print(f"Best by AIC: {df.loc[df['AIC'].idxmin(), 'Model']} (lower is better)")
    print(f"Best by BIC: {df.loc[df['BIC'].idxmin(), 'Model']} (lower is better)")


def display_linear_comparison(linear_models_info):
    """
    Display linear models comparison with AIC, BIC, Adjusted R2.
    
    Args:
        linear_models_info: List of dicts with model info
    """
    from evaluation import calculate_adjusted_r2, calculate_aic, calculate_bic
    
    print("\n" + "="*80)
    print("LINEAR MODELS COMPARISON")
    print("="*80)
    
    results = []
    for info in linear_models_info:
        r2_adj = calculate_adjusted_r2(info['r2_train'], info['n_train'], info['n_vars'])
        aic = calculate_aic(info['n_test'], info['mse_test'], info['n_vars'])
        bic = calculate_bic(info['n_test'], info['mse_test'], info['n_vars'])
        
        results.append({
            'Model': info['name'],
            'Variables': info['n_vars'],
            'R2 Test': info['r2_test'],
            'Adj R2': r2_adj,
            'AIC': aic,
            'BIC': bic
        })
    
    df = pd.DataFrame(results)
    
    print("\n" + df.to_string(index=False))
    print("="*80)
    
    print("\nNote: Lower AIC/BIC = Better. Higher R2/Adj R2 = Better.")
    print("      M3 (Lasso) applies regularization to M2b (Lags) variables.")
    
    print(f"\nBest by R2 Test: {df.loc[df['R2 Test'].idxmax(), 'Model']}")
    print(f"Best by Adj R2: {df.loc[df['Adj R2'].idxmax(), 'Model']}")
    print(f"Best by AIC: {df.loc[df['AIC'].idxmin(), 'Model']}")
    print(f"Best by BIC: {df.loc[df['BIC'].idxmin(), 'Model']}")


def prepare_data_splits(df_features):
    """
    Prepare all data splits for different models.
    
    Returns:
        dict with all train/test splits
    """
    from modeling import create_lagged_features, create_polynomial_features
    
    df_lags = create_lagged_features(df_features, ['X1', 'X2', 'X3', 'X4'], lag=1)
    
    train = df_lags.loc[:'2023-12-31']
    test = df_lags.loc['2024-01-31':]
    
    data = {
        'X_train': train[['X1', 'X2', 'X3', 'X4']].values,
        'y_train': train['Y'].values,
        'X_test': test[['X1', 'X2', 'X3', 'X4']].values,
        'y_test': test['Y'].values,
        'X_train_lags': train[['X1', 'X2', 'X3', 'X4', 'X1_lag1', 'X2_lag1', 'X3_lag1', 'X4_lag1']].values,
        'X_test_lags': test[['X1', 'X2', 'X3', 'X4', 'X1_lag1', 'X2_lag1', 'X3_lag1', 'X4_lag1']].values,
        'y_train_lags': train['Y'].values,
        'y_test_lags': test['Y'].values,
        'X_train_sel': train[['X1', 'X2', 'X3', 'X4', 'X3_lag1']].values,
        'X_test_sel': test[['X1', 'X2', 'X3', 'X4', 'X3_lag1']].values,
        'train_size': len(train),
        'test_size': len(test),
        'df_lags': df_lags
    }
    
    from modeling import create_polynomial_features
    data['X_train_poly'] = create_polynomial_features(data['X_train'], degree=2)
    data['X_test_poly'] = create_polynomial_features(data['X_test'], degree=2)
    
    return data


def train_all_models(data):
    """
    Train all models and return results.
    
    Args:
        data: dict from prepare_data_splits()
    
    Returns:
        models, results, predictions
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    from modeling import fit_linear_model, fit_lasso, fit_random_forest, fit_gradient_boosting
    from evaluation import evaluate_model
    
    models = {}
    results = {}
    predictions = {}
    
    print("  - Training M1: Simple Linear...")
    models['M1'] = fit_linear_model(data['X_train'], data['y_train'])
    predictions['M1'] = models['M1'].predict(data['X_test'])
    results['M1'] = evaluate_model(data['y_test'], predictions['M1'])
    results['M1']['n_vars'] = 4
    
    print("  - Training M2a: Polynomial...")
    models['M2a'] = fit_linear_model(data['X_train_poly'], data['y_train'])
    predictions['M2a'] = models['M2a'].predict(data['X_test_poly'])
    results['M2a'] = evaluate_model(data['y_test'], predictions['M2a'])
    results['M2a']['n_vars'] = 6
    
    print("  - Training M2b: Linear + Lags...")
    models['M2b'] = fit_linear_model(data['X_train_lags'], data['y_train_lags'])
    predictions['M2b'] = models['M2b'].predict(data['X_test_lags'])
    results['M2b'] = evaluate_model(data['y_test_lags'], predictions['M2b'])
    results['M2b']['n_vars'] = 8
    
    print("  - Training M3: Lasso + Lags...")
    models['M3'] = fit_lasso(data['X_train_lags'], data['y_train_lags'])
    predictions['M3'] = models['M3'].predict(data['X_test_lags'])
    results['M3'] = evaluate_model(data['y_test_lags'], predictions['M3'])
    results['M3']['n_vars'] = int(np.sum(models['M3'].coef_ != 0))
    
    print("  - Training M4: Random Forest...")
    models['M4'] = fit_random_forest(data['X_train_sel'], data['y_train_lags'], n_estimators=30, max_depth=4)
    predictions['M4'] = models['M4'].predict(data['X_test_sel'])
    results['M4'] = evaluate_model(data['y_test_lags'], predictions['M4'])
    results['M4']['n_vars'] = 5
    
    print("  - Training M5: Gradient Boosting...")
    models['M5'] = fit_gradient_boosting(data['X_train_sel'], data['y_train_lags'], n_estimators=50, learning_rate=0.05, max_depth=3)
    predictions['M5'] = models['M5'].predict(data['X_test_sel'])
    results['M5'] = evaluate_model(data['y_test_lags'], predictions['M5'])
    results['M5']['n_vars'] = 5
    
    return models, results, predictions


def save_all_results(models, results, predictions, data):
    """Save all results to files."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn import metrics
    from evaluation import calculate_adjusted_r2, calculate_aic, calculate_bic
    
    Path('06_results').mkdir(exist_ok=True)
    
    y_pred_m1_train = models['M1'].predict(data['X_train'])
    y_pred_m2a_train = models['M2a'].predict(data['X_train_poly'])
    y_pred_m2b_train = models['M2b'].predict(data['X_train_lags'])
    y_pred_m3_train = models['M3'].predict(data['X_train_lags'])
    
    r2_train_m1 = metrics.r2_score(data['y_train'], y_pred_m1_train)
    r2_train_m2a = metrics.r2_score(data['y_train'], y_pred_m2a_train)
    r2_train_m2b = metrics.r2_score(data['y_train_lags'], y_pred_m2b_train)
    r2_train_m3 = metrics.r2_score(data['y_train_lags'], y_pred_m3_train)
    
    linear_models_info = [
        {'name': 'M1: Simple', 'n_vars': 4, 'r2_test': results['M1']['R²'], 'r2_train': r2_train_m1, 'mse_test': results['M1']['MSE'], 'n_train': len(data['y_train']), 'n_test': len(data['y_test'])},
        {'name': 'M2a: Polynomial', 'n_vars': 6, 'r2_test': results['M2a']['R²'], 'r2_train': r2_train_m2a, 'mse_test': results['M2a']['MSE'], 'n_train': len(data['y_train']), 'n_test': len(data['y_test'])},
        {'name': 'M2b: Lags', 'n_vars': 8, 'r2_test': results['M2b']['R²'], 'r2_train': r2_train_m2b, 'mse_test': results['M2b']['MSE'], 'n_train': len(data['y_train_lags']), 'n_test': len(data['y_test_lags'])},
        {'name': 'M3: Lasso + Lags', 'n_vars': results['M3']['n_vars'], 'r2_test': results['M3']['R²'], 'r2_train': r2_train_m3, 'mse_test': results['M3']['MSE'], 'n_train': len(data['y_train_lags']), 'n_test': len(data['y_test_lags'])}
    ]
    
    linear_comparison_list = []
    for info in linear_models_info:
        linear_comparison_list.append({
            'Model': info['name'],
            'Variables': info['n_vars'],
            'R2_Test': info['r2_test'],
            'Adj_R2': calculate_adjusted_r2(info['r2_train'], info['n_train'], info['n_vars']),
            'AIC': calculate_aic(info['n_test'], info['mse_test'], info['n_vars']),
            'BIC': calculate_bic(info['n_test'], info['mse_test'], info['n_vars'])
        })
    linear_comparison_df = pd.DataFrame(linear_comparison_list)
    linear_comparison_df.to_csv('06_results/linear_comparison.csv', index=False)
    
    # All models comparison
    comparison = pd.DataFrame({
        'Model': ['M1: Simple', 'M2a: Polynomial', 'M2b: Lags', 'M3: Lasso+Lags', 'M4: Random Forest', 'M5: Gradient Boosting'],
        'Type': ['Linear', 'Linear', 'Linear', 'Linear (Regularized)', 'Non-Linear', 'Non-Linear'],
        'Variables': [results['M1']['n_vars'], results['M2a']['n_vars'], results['M2b']['n_vars'], results['M3']['n_vars'], results['M4']['n_vars'], results['M5']['n_vars']],
        'R2 Test': [results['M1']['R²'], results['M2a']['R²'], results['M2b']['R²'], results['M3']['R²'], results['M4']['R²'], results['M5']['R²']],
        'RMSE': [results['M1']['RMSE'], results['M2a']['RMSE'], results['M2b']['RMSE'], results['M3']['RMSE'], results['M4']['RMSE'], results['M5']['RMSE']],
        'MAE': [results['M1']['MAE'], results['M2a']['MAE'], results['M2b']['MAE'], results['M3']['MAE'], results['M4']['MAE'], results['M5']['MAE']]
    })
    comparison.to_csv('06_results/model_comparison.csv', index=False)
    
    # Best model
    best_idx = comparison['R2 Test'].idxmax()
    best_model = comparison.loc[best_idx, 'Model']
    best_r2 = comparison.loc[best_idx, 'R2 Test']
    
    with open('06_results/best_model.txt', 'w') as f:
        f.write(f"Best Model: {best_model}\n")
        f.write(f"R2 Test: {best_r2:.4f}\n")
        f.write(f"RMSE: {comparison.loc[best_idx, 'RMSE']:.6f}\n")
        f.write(f"MAE: {comparison.loc[best_idx, 'MAE']:.6f}\n")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(comparison)), comparison['R2 Test'], color='blue')
    plt.xticks(range(len(comparison)), ['M1', 'M2a', 'M2b', 'M3', 'M4', 'M5'])
    plt.ylabel('R2 Test')
    plt.title('Model Comparison: R2 Test Performance')
    plt.axhline(y=0, linestyle='--')
    plt.grid()
    plt.tight_layout()
    plt.savefig('05_figures/model_comparison.png')
    
    return linear_models_info, comparison, best_model, best_r2

