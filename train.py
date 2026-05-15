import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

print("=" * 80)
print("STUDENT DROPOUT PREDICTION - MODEL TRAINING PIPELINE")
print("=" * 80)

# ==================== STEP 1: LOAD DATA ====================
print("\n[STEP 1] Loading Data...")
file_name = 'students_dropout_academic_success.csv'
try:
    df = pd.read_csv(file_name)
    print(f"✓ Successfully loaded '{file_name}'")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"✗ Error: The file '{file_name}' was not found.")
    exit(1)

# ==================== STEP 2: DATA CLEANING ====================
print("\n[STEP 2] Data Cleaning...")

# Remove duplicates
duplicates = df.duplicated().sum()
print(f"  - Duplicate rows found: {duplicates}")
df = df.drop_duplicates()

# Handle missing values
print(f"  - Missing values before imputation:")
missing_before = df.isnull().sum()
if missing_before.sum() > 0:
    print(f"    {missing_before[missing_before > 0].to_dict()}")
else:
    print(f"    None")

numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print(f"  - Missing values after imputation: {df.isnull().sum().sum()}")
print(f"✓ Data cleaned successfully")

# ==================== STEP 3: TARGET ENCODING ====================
print("\n[STEP 3] Target Variable Encoding...")
print(f"  - Unique target values: {df['target'].unique()}")

label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])
print(f"  - Class mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
print(f"✓ Target variable encoded successfully")

# ==================== STEP 4: FEATURE SEPARATION ====================
print("\n[STEP 4] Feature Separation...")
X = df.drop('target', axis=1)
y = df['target']
print(f"  - Features (X): {X.shape}")
print(f"  - Target (y): {y.shape}")
print(f"  - Target distribution:")
for label, count in zip(label_encoder.classes_, np.bincount(y)):
    print(f"    {label}: {count} samples")
print(f"✓ Features separated successfully")

# ==================== STEP 5: TRAIN-TEST SPLIT ====================
print("\n[STEP 5] Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"  - Training set: {X_train.shape}")
print(f"  - Testing set: {X_test.shape}")
print(f"✓ Data split successfully (80-20)")

# ==================== STEP 6: FEATURE SCALING ====================
print("\n[STEP 6] Feature Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  - Scaling method: StandardScaler")
print(f"  - Training data scaled: {X_train_scaled.shape}")
print(f"✓ Features scaled successfully")

# ==================== STEP 7: CLASS BALANCING (SMOTE) ====================
print("\n[STEP 7] Class Balancing with SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"  - Before SMOTE:")
for label in label_encoder.classes_:
    count = (y_train == label_encoder.transform([label])[0]).sum()
    print(f"    {label}: {count}")

print(f"  - After SMOTE:")
for idx, label in enumerate(label_encoder.classes_):
    count = (y_train_resampled == idx).sum()
    print(f"    {label}: {count}")
print(f"✓ Classes balanced successfully")

# ==================== STEP 8: TRAIN MULTIPLE MODELS ====================
print("\n[STEP 8] Training Multiple Models...")
models = {
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Extra Trees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"  - Training {name}...", end=" ")
    model.fit(X_train_resampled, y_train_resampled)
    predictions = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    
    results.append([name, accuracy, f1])
    trained_models[name] = model
    
    print(f"✓ (Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f})")

# ==================== STEP 9: MODEL COMPARISON ====================
print("\n[STEP 9] Model Comparison...")
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1 Score'])
results_df = results_df.sort_values(by='Accuracy', ascending=False)

print("\nModel Performance Ranking:")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
print(f"\n✓ Best Model: {best_model_name}")

# ==================== STEP 10: TRAIN FINAL BEST MODEL ====================
print("\n[STEP 10] Training Final Best Model (CatBoost)...")
best_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function='MultiClass',
    verbose=0,
    random_state=42
)

best_model.fit(X_train_resampled, y_train_resampled)
predictions = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

print(f"\n  - Final Accuracy: {accuracy:.4f}")
print(f"  - Final F1-Score: {f1:.4f}")

# ==================== STEP 11: DETAILED EVALUATION ====================
print("\n[STEP 11] Detailed Model Evaluation...")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)

# ==================== STEP 12: SAVE MODELS ====================
print("\n[STEP 12] Saving Models and Preprocessing Objects...")
os.makedirs('models', exist_ok=True)

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print(f"  ✓ Best model saved: models/best_model.pkl")
print(f"  ✓ Scaler saved: models/scaler.pkl")
print(f"  ✓ Label encoder saved: models/label_encoder.pkl")

# ==================== STEP 13: VERIFY SAVED MODELS ====================
print("\n[STEP 13] Verifying Saved Models...")
try:
    loaded_model = joblib.load('models/best_model.pkl')
    loaded_scaler = joblib.load('models/scaler.pkl')
    loaded_encoder = joblib.load('models/label_encoder.pkl')
    print("  ✓ All models loaded successfully!")
except Exception as e:
    print(f"  ✗ Error loading models: {e}")

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nYou can now run the Streamlit app:")
print("  $ streamlit run app.py")
print("\n" + "=" * 80)
