import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fall back to LightGBM or GradientBoosting if not available
try:
    import xgboost as xgb
    MODEL_TYPE = 'XGBoost'
    print(" Using XGBoost")
except Exception as e:
    print(f"  XGBoost not available: {str(e)[:100]}")
    try:
        import lightgbm as lgb
        MODEL_TYPE = 'LightGBM'
        print(" Using LightGBM as alternative")
    except:
        from sklearn.ensemble import GradientBoostingRegressor
        MODEL_TYPE = 'GradientBoosting'
        print(" Using scikit-learn GradientBoosting as alternative")

print(f"{MODEL_TYPE} PREDICTION: STRESS & ANXIETY LEVELS")


df = pd.read_csv('/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/ml_ready_dataset_transformed.csv')

print(f" Dataset loaded: {df.shape}")
print(f"  Rows: {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")

# PREPARE FEATURES AND TARGETS


# Separate features and targets
X = df.drop(['stress_level', 'anxiety_level'], axis=1)
y_stress = df['stress_level']
y_anxiety = df['anxiety_level']

print(f" Features shape: {X.shape}")
print(f" Stress target shape: {y_stress.shape}")
print(f" Anxiety target shape: {y_anxiety.shape}")

# Check for missing data
missing_cols = X.columns[X.isna().any()].tolist()
if missing_cols:
    print(f"\n  Missing data found in {len(missing_cols)} columns")
    for col in missing_cols[:5]:
        pct = (X[col].isna().sum() / len(X)) * 100
        print(f"  • {col}: {pct:.1f}% missing")
    if len(missing_cols) > 5:
        print(f"  ... and {len(missing_cols) - 5} more")

# HANDLE MISSING DATA


# Use median imputation for numerical features
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print(f" Missing data imputed using median strategy")
print(f"  Remaining missing values: {X_imputed.isna().sum().sum()}")



# Split for stress prediction
X_train_stress, X_test_stress, y_train_stress, y_test_stress = train_test_split(
    X_imputed, y_stress, test_size=0.2, random_state=42, stratify=None
)

# Split for anxiety prediction
X_train_anxiety, X_test_anxiety, y_train_anxiety, y_test_anxiety = train_test_split(
    X_imputed, y_anxiety, test_size=0.2, random_state=42, stratify=None
)

print(" Stress prediction sets:")
print(f"  Training: {X_train_stress.shape[0]} samples")
print(f"  Testing: {X_test_stress.shape[0]} samples")

print(" Anxiety prediction sets:")
print(f"  Training: {X_train_anxiety.shape[0]} samples")
print(f"  Testing: {X_test_anxiety.shape[0]} samples")

# TRAIN XGBOOST MODELS


# Create models based on what's available
if MODEL_TYPE == 'XGBoost':
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    model_stress = xgb.XGBRegressor(**params)
    model_anxiety = xgb.XGBRegressor(**params)
    
elif MODEL_TYPE == 'LightGBM':
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    model_stress = lgb.LGBMRegressor(**params)
    model_anxiety = lgb.LGBMRegressor(**params)
    
else:  # GradientBoosting
    # Scikit-learn GradientBoosting parameters
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'random_state': 42
    }
    model_stress = GradientBoostingRegressor(**params)
    model_anxiety = GradientBoostingRegressor(**params)

# Train stress model
model_stress.fit(X_train_stress, y_train_stress)
print("   Stress model trained")

# Train anxiety model
model_anxiety.fit(X_train_anxiety, y_train_anxiety)
print("   Anxiety model trained")



# Stress predictions
y_pred_stress_train = model_stress.predict(X_train_stress)
y_pred_stress_test = model_stress.predict(X_test_stress)

# Anxiety predictions
y_pred_anxiety_train = model_anxiety.predict(X_train_anxiety)
y_pred_anxiety_test = model_anxiety.predict(X_test_anxiety)

print(" Predictions completed")

# EVALUATE MODELS


def evaluate_regression(y_true, y_pred, dataset_name):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Dataset': dataset_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

# Evaluate stress model
stress_train_metrics = evaluate_regression(y_train_stress, y_pred_stress_train, 'Train')
stress_test_metrics = evaluate_regression(y_test_stress, y_pred_stress_test, 'Test')

# Evaluate anxiety model
anxiety_train_metrics = evaluate_regression(y_train_anxiety, y_pred_anxiety_train, 'Train')
anxiety_test_metrics = evaluate_regression(y_test_anxiety, y_pred_anxiety_test, 'Test')

# Display results
print("\n" + "="*80)
print("STRESS PREDICTION RESULTS")
stress_results = pd.DataFrame([stress_train_metrics, stress_test_metrics])
print(stress_results.to_string(index=False))

print("\n" + "="*80)
print("ANXIETY PREDICTION RESULTS")
anxiety_results = pd.DataFrame([anxiety_train_metrics, anxiety_test_metrics])
print(anxiety_results.to_string(index=False))

# Check for overfitting
print("\n" + "="*80)
print("OVERFITTING CHECK")
stress_overfit = stress_train_metrics['R²'] - stress_test_metrics['R²']
anxiety_overfit = anxiety_train_metrics['R²'] - anxiety_test_metrics['R²']

print(f"Stress model:")
print(f"  Train R²: {stress_train_metrics['R²']:.4f}")
print(f"  Test R²: {stress_test_metrics['R²']:.4f}")
print(f"  Difference: {stress_overfit:.4f} {'  (overfitting)' if stress_overfit > 0.1 else ' (good)'}")

print(f"\nAnxiety model:")
print(f"  Train R²: {anxiety_train_metrics['R²']:.4f}")
print(f"  Test R²: {anxiety_test_metrics['R²']:.4f}")
print(f"  Difference: {anxiety_overfit:.4f} {'  (overfitting)' if anxiety_overfit > 0.1 else ' (good)'}")

# FEATURE IMPORTANCE


# Get feature importance based on model type
if MODEL_TYPE == 'GradientBoosting':
    stress_importances = model_stress.feature_importances_
    anxiety_importances = model_anxiety.feature_importances_
else:
    stress_importances = model_stress.feature_importances_
    anxiety_importances = model_anxiety.feature_importances_

# Get feature importance for stress model
stress_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': stress_importances
}).sort_values('Importance', ascending=False)

# Get feature importance for anxiety model
anxiety_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': anxiety_importances
}).sort_values('Importance', ascending=False)

print("\n" + "="*80)
print("TOP 10 MOST IMPORTANT FEATURES FOR STRESS PREDICTION")
print(stress_importance.head(10).to_string(index=False))

print("\n" + "="*80)
print("TOP 10 MOST IMPORTANT FEATURES FOR ANXIETY PREDICTION")
print(anxiety_importance.head(10).to_string(index=False))

# VISUALIZATIONS

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Stress - Actual vs Predicted (Test)
ax1 = axes[0, 0]
ax1.scatter(y_test_stress, y_pred_stress_test, alpha=0.5, s=20)
ax1.plot([y_test_stress.min(), y_test_stress.max()], 
         [y_test_stress.min(), y_test_stress.max()], 
         'r--', lw=2)
ax1.set_xlabel('Actual Stress Level')
ax1.set_ylabel('Predicted Stress Level')
ax1.set_title(f'Stress: Actual vs Predicted (Test)\nR² = {stress_test_metrics["R²"]:.3f}')
ax1.grid(True, alpha=0.3)

# 2. Stress - Residuals
ax2 = axes[0, 1]
residuals_stress = y_test_stress - y_pred_stress_test
ax2.scatter(y_pred_stress_test, residuals_stress, alpha=0.5, s=20)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Stress Level')
ax2.set_ylabel('Residuals')
ax2.set_title('Stress: Residual Plot')
ax2.grid(True, alpha=0.3)

# 3. Stress - Feature Importance
ax3 = axes[0, 2]
top_10_stress = stress_importance.head(10)
ax3.barh(range(len(top_10_stress)), top_10_stress['Importance'])
ax3.set_yticks(range(len(top_10_stress)))
ax3.set_yticklabels(top_10_stress['Feature'], fontsize=8)
ax3.set_xlabel('Importance')
ax3.set_title('Top 10 Features for Stress')
ax3.invert_yaxis()

# 4. Anxiety - Actual vs Predicted (Test)
ax4 = axes[1, 0]
ax4.scatter(y_test_anxiety, y_pred_anxiety_test, alpha=0.5, s=20, color='orange')
ax4.plot([y_test_anxiety.min(), y_test_anxiety.max()], 
         [y_test_anxiety.min(), y_test_anxiety.max()], 
         'r--', lw=2)
ax4.set_xlabel('Actual Anxiety Level')
ax4.set_ylabel('Predicted Anxiety Level')
ax4.set_title(f'Anxiety: Actual vs Predicted (Test)\nR² = {anxiety_test_metrics["R²"]:.3f}')
ax4.grid(True, alpha=0.3)

# 5. Anxiety - Residuals
ax5 = axes[1, 1]
residuals_anxiety = y_test_anxiety - y_pred_anxiety_test
ax5.scatter(y_pred_anxiety_test, residuals_anxiety, alpha=0.5, s=20, color='orange')
ax5.axhline(y=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('Predicted Anxiety Level')
ax5.set_ylabel('Residuals')
ax5.set_title('Anxiety: Residual Plot')
ax5.grid(True, alpha=0.3)

# 6. Anxiety - Feature Importance
ax6 = axes[1, 2]
top_10_anxiety = anxiety_importance.head(10)
ax6.barh(range(len(top_10_anxiety)), top_10_anxiety['Importance'], color='orange')
ax6.set_yticks(range(len(top_10_anxiety)))
ax6.set_yticklabels(top_10_anxiety['Feature'], fontsize=8)
ax6.set_xlabel('Importance')
ax6.set_title('Top 10 Features for Anxiety')
ax6.invert_yaxis()

plt.tight_layout()
output_plot = f'/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/anova/{MODEL_TYPE.lower()}_results.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.show()

print(f" Saved visualization: {output_plot}")


print("\n" + "="*80)
print("SAVING RESULTS")

base_path = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/anova/'

# Save metrics
metrics_df = pd.DataFrame({
    'Model': ['Stress', 'Stress', 'Anxiety', 'Anxiety'],
    'Dataset': ['Train', 'Test', 'Train', 'Test'],
    'MSE': [stress_train_metrics['MSE'], stress_test_metrics['MSE'],
            anxiety_train_metrics['MSE'], anxiety_test_metrics['MSE']],
    'RMSE': [stress_train_metrics['RMSE'], stress_test_metrics['RMSE'],
             anxiety_train_metrics['RMSE'], anxiety_test_metrics['RMSE']],
    'MAE': [stress_train_metrics['MAE'], stress_test_metrics['MAE'],
            anxiety_train_metrics['MAE'], anxiety_test_metrics['MAE']],
    'R²': [stress_train_metrics['R²'], stress_test_metrics['R²'],
           anxiety_train_metrics['R²'], anxiety_test_metrics['R²']]
})
metrics_df.to_csv(f'{base_path}xgboost_metrics.csv', index=False)
print(f" Saved: xgboost_metrics.csv")

# Save feature importance
stress_importance.to_csv(f'{base_path}xgboost_feature_importance_stress.csv', index=False)
anxiety_importance.to_csv(f'{base_path}xgboost_feature_importance_anxiety.csv', index=False)
print(f" Saved: xgboost_feature_importance_stress.csv")
print(f" Saved: xgboost_feature_importance_anxiety.csv")

# Save predictions
predictions_stress = pd.DataFrame({
    'Actual_Stress': y_test_stress.values,
    'Predicted_Stress': y_pred_stress_test
})
predictions_stress.to_csv(f'{base_path}xgboost_predictions_stress.csv', index=False)

predictions_anxiety = pd.DataFrame({
    'Actual_Anxiety': y_test_anxiety.values,
    'Predicted_Anxiety': y_pred_anxiety_test
})
predictions_anxiety.to_csv(f'{base_path}xgboost_predictions_anxiety.csv', index=False)
print(f" Saved: xgboost_predictions_stress.csv")
print(f" Saved: xgboost_predictions_anxiety.csv")

# SUMMARY

print("\n" + "="*80)

print("\n KEY FINDINGS:")
print(f"\nStress Prediction:")
print(f"  • Test R² = {stress_test_metrics['R²']:.3f} ({stress_test_metrics['R²']*100:.1f}% variance explained)")
print(f"  • Test RMSE = {stress_test_metrics['RMSE']:.2f}")
print(f"  • Test MAE = {stress_test_metrics['MAE']:.2f}")
print(f"  • Top predictor: {stress_importance.iloc[0]['Feature']}")

print(f"\nAnxiety Prediction:")
print(f"  • Test R² = {anxiety_test_metrics['R²']:.3f} ({anxiety_test_metrics['R²']*100:.1f}% variance explained)")
print(f"  • Test RMSE = {anxiety_test_metrics['RMSE']:.2f}")
print(f"  • Test MAE = {anxiety_test_metrics['MAE']:.2f}")
print(f"  • Top predictor: {anxiety_importance.iloc[0]['Feature']}")

print("\n NEXT STEPS:")
print("  1. Fine-tune hyperparameters with GridSearchCV")
print("  2. Try ensemble methods (combine multiple models)")
print("  3. Perform cross-validation for more robust estimates")
print("  4. Consider feature engineering based on importance")
print("  5. Try classification approach (Low/Medium/High categories)")

print("\n" + "="*80)