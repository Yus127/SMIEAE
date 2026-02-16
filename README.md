# SMIEAE

Stress Monitoring in Educational Academic Environments — a data pipeline for predicting stress and anxiety levels in university students using Fitbit physiological data, daily questionnaires, and environmental sensors (ESFM).

## Project Overview

This project collects and processes multi-modal data from university students across two institutions and multiple courses:

- **University 1**: Courses A1 (users 1-10), A2 (users 11-20), C (users 36-55)
- **University 2**: Course B (users 21-35)

Data sources include:
- **Fitbit wearables**: heart rate, HRV, SpO2, respiratory rate, sleep stages, activity levels, steps
- **Daily questionnaires**: self-reported stress and anxiety levels (slider-based)
- **ESFM environmental sensors**: temperature, CO2, pressure, occupancy counts from classroom sensors

The pipeline cleans raw data, creates time-windowed features aligned with questionnaire responses, and trains classification models to predict stress/anxiety levels.

## Directory Structure

```
SMIEAE/
├── pre_process_data/          # Raw data cleaning and EDA
│   ├── daily_questions/       # Questionnaire data processing
│   ├── fitbit/                # Fitbit wearable data processing
│   └── esfm/                  # Environmental sensor data processing
├── pre_process_algorithms/    # Feature engineering and statistical analysis
│   └── analyze_data/          # Dataset completeness analysis
├── add_exams/                 # Academic calendar feature engineering
├── create_windows_5_10_mins/  # 5-min and 10-min window extraction
├── create_windows_30_60_mins/ # 30-min and 60-min window extraction
├── ml/                        # Machine learning models
│   ├── whole_dataset/         # Models on daily aggregated data (random split)
│   │   └── timeseries/        # Same models with time-series split
│   ├── 5_10_window/           # Models on 5/10-min windowed data
│   ├── 30_60_window/          # Models on 30/60-min windowed data
│   ├── anomaly_detection/     # Anomaly detection algorithms
│   └── clusters/              # K-Means clustering analysis
├── requirements.txt
└── README.md
```

## Data Pipeline

The pipeline runs sequentially through the following stages:

### Stage 1: Raw Data Cleaning

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `pre_process_data/daily_questions/clean_daily_q.py` | Cleans questionnaire export: parses timestamps (no timezone offset), applies 2AM day-assignment rule, drops unused columns | `data/original_data/daily_questions *.csv` | `data/data_clean/daily_questions_cleaned.csv` |
| `pre_process_data/fitbit/clean_fitbit.py` | Consolidates per-user Fitbit JSON/CSV exports (heart rate, HRV, SpO2, sleep, steps, activity) into single CSVs per user, merging multiple data sources on timestamp | `data/original_data/fitbit/<user>/` folders | `data/original_data/fitbit/consolidated_output/user_*_consolidated.csv` |
| `pre_process_data/fitbit/clean_fitbit2.py` | Separates consolidated Fitbit data into daily summaries (sleep metrics, daily aggregations) and time-series data. Extracts detailed sleep stage metrics, computes daily HR/HRV/SpO2 aggregations, filters sensor errors | `consolidated_output/user_*_consolidated.csv` | `data/data_clean/fitbit/*_daily_summary.csv`, `*_timeseries.csv` |
| `pre_process_data/esfm/process_sensors_ESFM.py` | Categorizes and combines ESFM classroom sensor CSV files by sensor type | Raw ESFM sensor CSVs | `data/data_clean/ESFM/combined_ESFM_*.csv` |

### Stage 1b: Exploratory Data Analysis

| Script | Purpose |
|--------|---------|
| `pre_process_data/daily_questions/exploratory_daily_q.py` | EDA on questionnaire data: distributions, time series, weekday patterns, stress vs anxiety scatter |
| `pre_process_data/daily_questions/exploratory_classes_stress.py` | Per-course stress/anxiety averages and daily trends |
| `pre_process_data/daily_questions/stadistics_daily_q.py` | Date continuity analysis: checks for missing days per user |
| `pre_process_data/fitbit/exploratory_fitbit.py` | EDA on raw Fitbit consolidated data: column types, distributions, date ranges |
| `pre_process_data/fitbit/eda_data_clean_fitbit.py` | EDA on cleaned daily summary data: variable categories, completeness |
| `pre_process_data/fitbit/tmp.py` | Data completeness report across all user files |
| `pre_process_data/esfm/exploratory_ESFM.py` | Filters ESFM data to Class B schedule (Mon/Tue/Thu 09:30-11:00) and produces exploratory stats |

### Stage 2: Feature Engineering on Daily Data

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `pre_process_algorithms/join_csv.py` | Joins Fitbit daily summaries with cleaned questionnaires on (userid, date). Excludes users 39 and 54. Creates stress/anxiety level columns from self-report scores | `data/data_clean/fitbit/*_daily_summary.csv` + `daily_questions_cleaned.csv` | `data/data_clean/csv_joined/combined_daily_data.csv` |
| `pre_process_algorithms/add_log_transforms.py` | Adds log1p transforms for columns with skewness > 1.0. Skips deviation columns, sentinel values, and target variables | `combined_daily_data.csv` | `combined_daily_data_with_log_transforms.csv` |
| `add_exams/add_columns_exam.py` | Adds academic calendar features: exam period flags, days/weeks to next and since last exam, Semana Santa flag, exam proximity categories. University-specific exam schedules | `combined_daily_data_with_log_transforms.csv` | `data_with_exam_features.csv` |

### Stage 2b: Statistical Analysis

| Script | Purpose |
|--------|---------|
| `pre_process_algorithms/correlation_fitbit.py` | Correlation matrix analysis of Fitbit daily summary variables |
| `pre_process_algorithms/correlation_stress_anxiety.py` | Correlation analysis between Fitbit features and stress/anxiety targets |
| `pre_process_algorithms/anova_stress_anxiety.py` | ANOVA / XGBoost feature importance analysis for stress and anxiety prediction |
| `pre_process_algorithms/dbscan_daily_q.py` | DBSCAN clustering on questionnaire stress/anxiety responses |
| `pre_process_algorithms/dbscan_fitbit.py` | DBSCAN outlier detection on Fitbit health variables with forward-fill imputation |
| `pre_process_algorithms/analyze_data/analyze_all_csvs.py` | Completeness and volume analysis of all ML-ready CSVs |

### Stage 3: Time-Windowed Feature Extraction

These scripts extract Fitbit physiological data from specific time windows **before** each questionnaire response, then create ML-ready datasets.

#### 5-min / 10-min Windows (`create_windows_5_10_mins/`)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `filter_questionaty_fitbit.py` | For each questionnaire response, extracts Fitbit records from 5-min and 10-min windows before the response timestamp. Produces both detailed (raw records) and aggregated (mean/std/min/max/median per feature) versions | `consolidated_output/*.csv` + `daily_questions_cleaned.csv` | `data/processed/5_10min/fitbit_{5,10}min_window_{detailed,aggregated}.csv` |
| `create_datatset.py` | Converts aggregated window data into ML-ready datasets: removes low-completeness features, encodes targets, creates train/test metadata | `data/processed/5_10min/*_aggregated.csv` | `data/ml_ready/ml_ready_{5,10}min_window.csv` |
| `add_steps_examns.py` | Enriches ML-ready datasets with daily step counts, exam period features, and course/university metadata | `data/ml_ready/*.csv` + `consolidated_output/*.csv` | `data/ml_ready/enriched/ml_ready_*_enriched.csv` |
| `analyze_variable_complet.py` | Analyzes feature completeness across enriched datasets | | |
| `median_mean_answers.py` | Analyzes median and mean of questionnaire responses and response times | | |

#### 30-min / 60-min Windows (`create_windows_30_60_mins/`)

Same pipeline as 5/10-min but with 30-minute and 1-hour windows:

| Script | Purpose |
|--------|---------|
| `filter_questionaty_fitbit.py` | Extracts 30-min and 60-min pre-response windows |
| `create_datatset.py` | Creates ML-ready datasets from 30/60-min aggregated data |
| `add_steps_examns.py` | Enriches with steps, exams, and course metadata |
| `analyze_variable_complet.py` | Feature completeness analysis |

### Stage 4: Machine Learning

All ML scripts predict **stress** and **anxiety** as classification targets (binary or 3-class based on percentile splits). Models used: Logistic Regression, Random Forest, XGBoost, SVM, Naive Bayes, and ensemble voting classifiers.

#### Whole Dataset (`ml/whole_dataset/`)

Uses daily aggregated data (`data_with_exam_features.csv`).

| Script | Approach | Split |
|--------|----------|-------|
| `approach1_ml_predict.py` | Multi-model comparison (5 classifiers), 3-class targets (p33/p67 split) | Random 70/15/15 |
| `approach2_clusters.py` | K-Means user clustering (K=2) on physiological profiles, then binary classification with cluster as feature | Random 70/15/15 |
| `approach3_pca_3.py` | PCA-based feature engineering with personal baselines and deviation features, strongly regularized 3-class classification | Random 70/15/15 |
| `approach4_lstm.py` | Bidirectional LSTM with 3-timestep windows, binary classification (median split) | Random 70/15/15 |

#### Whole Dataset with Time-Series Split (`ml/whole_dataset/timeseries/`)

Same 4 approaches but with **chronological time-series split** (ordered by date) to prevent temporal data leakage.

#### 5/10-min Windows (`ml/5_10_window/`)

| Script | Approach |
|--------|----------|
| `approach1_ml_predict.py` | Multi-model comparison on 5-min and 10-min enriched datasets |
| `approach2_clusters.py` | K-Means (K=5) + classification on windowed data |
| `approach3_pca_3.py` | PCA + regularized 3-class classification |
| `approach4_lstm.py` | Bi-LSTM on windowed physiological features |

#### 30/60-min Windows (`ml/30_60_window/`)

Same 4 approaches on 30-min and 60-min windowed data (K=3 for clustering).

#### Anomaly Detection (`ml/anomaly_detection/`)

| Script | Algorithm | Input | Output |
|--------|-----------|-------|--------|
| `isolation_forest_anomaly.py` | Isolation Forest | `data/ml_ready/enriched/ml_ready_*_enriched.csv` (5 datasets) | `results/anomaly_detection/` |
| `local_outlier_fact.py` | Local Outlier Factor | Same 5 enriched datasets | `results/anomaly_detection_lof/` |
| `onec_svm.py` | One-Class SVM | Same 5 enriched datasets | `results/anomaly_detection_ocsvm/` |

Each anomaly detection script produces per-window results CSVs, feature importance CSVs, analysis plots, and a summary CSV.

#### Clustering (`ml/clusters/`)

| Script | Purpose |
|--------|---------|
| `kmeans_whole_data.py` | K-Means (K=2) on daily aggregated data with PCA visualization |
| `kmeans_5_10.py` | K-Means (K=5) on 5/10-min windowed data |
| `kmeans_30_60.py` | K-Means (K=3) on 30/60-min windowed data |

#### Feature Importance

| Script | Purpose |
|--------|---------|
| `ml/shap_a.py` | SHAP-based feature importance analysis on combined windowed dataset |

## Key Features Used

**Fitbit Physiological:**
- Heart rate (mean, std, min, max)
- Heart rate variability (RMSSD, entropy, LF/HF ratio)
- SpO2 (blood oxygen saturation)
- Respiratory rate
- Sleep metrics (duration, efficiency, deep/light/REM/wake minutes, stage transitions, micro-awakenings)
- Daily steps and activity levels

**Contextual:**
- Exam period flags and proximity metrics
- University and course identifiers
- Day of week, Semana Santa (Easter break)

**Targets:**
- `stress_level`: derived from self-reported stress (slider)
- `anxiety_level`: derived from self-reported anxiety (slider)

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies: pandas, numpy, scikit-learn, xgboost, tensorflow/keras (for LSTM), shap, matplotlib, seaborn, statsmodels.

## Usage

Scripts are designed to run independently and sequentially through the pipeline stages. Each script has hardcoded input/output paths under the project's `data/` directory.

```bash
# Stage 1: Clean raw data
python pre_process_data/daily_questions/clean_daily_q.py
python pre_process_data/fitbit/clean_fitbit.py
python pre_process_data/fitbit/clean_fitbit2.py

# Stage 2: Join and engineer features
python pre_process_algorithms/join_csv.py
python pre_process_algorithms/add_log_transforms.py
python add_exams/add_columns_exam.py

# Stage 3: Create windowed datasets
python create_windows_5_10_mins/filter_questionaty_fitbit.py
python create_windows_5_10_mins/create_datatset.py
python create_windows_5_10_mins/add_steps_examns.py

python create_windows_30_60_mins/filter_questionaty_fitbit.py
python create_windows_30_60_mins/create_datatset.py
python create_windows_30_60_mins/add_steps_examns.py

# Stage 4: Train models (examples)
python ml/whole_dataset/approach1_ml_predict.py
python ml/5_10_window/approach1_ml_predict.py
python ml/30_60_window/approach1_ml_predict.py

# Anomaly detection
python ml/anomaly_detection/isolation_forest_anomaly.py
python ml/anomaly_detection/local_outlier_fact.py
python ml/anomaly_detection/onec_svm.py
```

## Data Directory Layout

```
data/
├── original_data/
│   ├── daily_questions *.csv           # Raw questionnaire export (semicolon-delimited)
│   └── fitbit/
│       ├── <user_folders>/             # Per-user Fitbit exports (JSON + CSV)
│       └── consolidated_output/        # Consolidated CSVs per user
├── data_clean/
│   ├── daily_questions_cleaned.csv
│   ├── fitbit/                         # Daily summaries + time-series per user
│   ├── csv_joined/                     # Merged Fitbit + questionnaire datasets
│   └── ESFM/                           # Cleaned environmental sensor data
├── processed/
│   └── 5_10min/                        # 5/10-min window extractions
├── ml_ready/
│   ├── ml_ready_*_window.csv           # ML-ready datasets
│   └── enriched/                       # Enriched with steps + exams
└── results/
    ├── whole_dataset/                  # Whole-dataset model results
    ├── 5_10_dataset/                   # 5/10-min window model results
    ├── 30_60_dataset/                  # 30/60-min window model results
    ├── anomaly_detection/              # Isolation Forest results
    ├── anomaly_detection_lof/          # LOF results
    ├── anomaly_detection_ocsvm/        # One-Class SVM results
    └── clustering_analysis/            # K-Means results
```
