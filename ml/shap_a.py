"""
SHAP (SHapley Additive exPlanations) Analysis
This script performs SHAP analysis on your dataset to explain model predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import shap
import os

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ==================== OUTPUT DIRECTORY ====================
# Create output directory if it doesn't exist
output_dir = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/shap'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# ==================== STEP 1: LOAD YOUR DATA ====================
# Loading the dataset
df = pd.read_csv('/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/combined_daily_data_with_log_transforms.csv')

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# ==================== STEP 2: DATA PREPROCESSING ====================
# Drop columns that should not be included in the analysis
columns_to_drop = ['anxiety_level', 'userid']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
print(f"\nDropped columns: {[col for col in columns_to_drop if col in df.columns]}")

# Handle missing values
df = df.dropna()  # Or use df.fillna(method='ffill') or df.fillna(df.mean())

# Separate features and target
# Target column is set to 'stress and anxiety'
target_column = 'stress_level'
X = df.drop(columns=[target_column])
y = df[target_column]

# Handle categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Determine if it's a classification or regression problem
is_classification = y.dtype == 'object' or len(y.unique()) < 20

if is_classification and y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

print(f"\nProblem Type: {'Classification' if is_classification else 'Regression'}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {X.columns.tolist()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optional: Scale features (recommended for some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to preserve column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# ==================== STEP 3: TRAIN MODEL ====================
if is_classification:
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"\nModel Accuracy: {score:.4f}")
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"\nModel R² Score: {score:.4f}")

# ==================== STEP 4: SHAP ANALYSIS ====================
print("\nCalculating SHAP values... (this may take a few minutes)")

# Create SHAP explainer
# For tree-based models (RandomForest, XGBoost, LightGBM), use TreeExplainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for test set
# Use a sample if dataset is too large (for faster computation)
sample_size = min(100, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

print("SHAP values calculated successfully!")

# ==================== STEP 5: VISUALIZATIONS ====================
print("\nGenerating SHAP visualizations...")

# 1. Summary Plot (Bar) - Feature Importance
plt.figure(figsize=(10, 8))
if is_classification and isinstance(shap_values, list):
    # For multi-class classification, use the first class
    shap.summary_plot(shap_values[0], X_test_sample, plot_type="bar", show=False)
else:
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_feature_importance.png'), dpi=300, bbox_inches='tight')
print("✓ Feature importance plot saved as 'shap_feature_importance.png'")
plt.close()

# 2. Summary Plot (Beeswarm) - Feature Impact
plt.figure(figsize=(10, 8))
if is_classification and isinstance(shap_values, list):
    shap.summary_plot(shap_values[0], X_test_sample, show=False)
else:
    shap.summary_plot(shap_values, X_test_sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
print("✓ Summary beeswarm plot saved as 'shap_summary_beeswarm.png'")
plt.close()

# 3. Dependence Plot for top 3 features
if is_classification and isinstance(shap_values, list):
    shap_vals_for_dependence = shap_values[0]
else:
    shap_vals_for_dependence = shap_values

# Get top features
mean_abs_shap = np.abs(shap_vals_for_dependence).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]
top_features = X_test_sample.columns[top_features_idx]

for i, feature in enumerate(top_features):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_vals_for_dependence, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shap_dependence_{feature}.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Dependence plot for '{feature}' saved")
    plt.close()

# 4. Waterfall Plot for a single prediction
plt.figure(figsize=(10, 8))
if is_classification and isinstance(shap_values, list):
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0][0],
            base_values=explainer.expected_value[0],
            data=X_test_sample.iloc[0],
            feature_names=X_test_sample.columns.tolist()
        ),
        show=False
    )
else:
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X_test_sample.iloc[0],
            feature_names=X_test_sample.columns.tolist()
        ),
        show=False
    )
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_waterfall_single.png'), dpi=300, bbox_inches='tight')
print("✓ Waterfall plot saved as 'shap_waterfall_single.png'")
plt.close()

# 5. Force Plot (save as HTML for interactivity)
if is_classification and isinstance(shap_values, list):
    force_plot = shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][0],
        X_test_sample.iloc[0],
        matplotlib=False
    )
else:
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_test_sample.iloc[0],
        matplotlib=False
    )
shap.save_html(os.path.join(output_dir, 'shap_force_plot.html'), force_plot)
print("✓ Interactive force plot saved as 'shap_force_plot.html'")

# ==================== STEP 6: EXPORT SHAP VALUES ====================
# Save SHAP values to CSV for further analysis
if is_classification and isinstance(shap_values, list):
    shap_df = pd.DataFrame(shap_values[0], columns=X_test_sample.columns)
else:
    shap_df = pd.DataFrame(shap_values, columns=X_test_sample.columns)

shap_df['prediction'] = model.predict(X_test_sample)
shap_df.to_csv(os.path.join(output_dir, 'shap_values.csv'), index=False)
print("✓ SHAP values exported to 'shap_values.csv'")

# Print feature importance based on mean absolute SHAP values
print("\n" + "="*60)
print("FEATURE IMPORTANCE (based on mean |SHAP value|)")
print("="*60)
feature_importance = pd.DataFrame({
    'feature': X_test_sample.columns,
    'importance': mean_abs_shap
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"{row['feature']:30s}: {row['importance']:.4f}")

print("\n" + "="*60)
print("SHAP Analysis Complete!")
print("="*60)
print(f"\nAll files saved to: {output_dir}")
print("\nGenerated files:")
print("  • shap_feature_importance.png - Feature importance bar plot")
print("  • shap_summary_beeswarm.png - Summary plot showing feature impacts")
print("  • shap_dependence_*.png - Dependence plots for top features")
print("  • shap_waterfall_single.png - Explanation for a single prediction")
print("  • shap_force_plot.html - Interactive force plot")
print("  • shap_values.csv - Raw SHAP values")