import pandas as pd
import joblib
import numpy as np

print("\n" + "="*70)
print("INSPECT TRAINING DATA & MODEL")
print("="*70)

# Load training data
try:
    df = pd.read_csv('data/sajojo_training_data.csv', header=None)
    
    print(f"\n📊 DATASET INFO:")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Total columns: {len(df.columns)}")
    print(f"  - Feature columns: {len(df.columns) - 1}")
    
    print(f"\n📈 CLASS DISTRIBUTION:")
    class_counts = df[0].value_counts()
    print(class_counts)
    
    print(f"\n🔢 DETAILED STATS:")
    for class_name, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  - {class_name}: {count} ({percentage:.1f}%)")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\n🔄 DUPLICATE ROWS: {duplicates}")
    
    # Check data quality
    print(f"\n✅ DATA QUALITY:")
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    print(f"  - Data types: {df.dtypes.value_counts().to_dict()}")
    
    # Sample data
    print(f"\n📝 SAMPLE DATA (first 3 rows):")
    print(df.head(3))
    
except FileNotFoundError:
    print("\n❌ File 'data/sajojo_training_data.csv' not found!")
    print("   Run retrain_sajojo_complete.py first")
    exit()

# Inspect trained model
print("\n" + "="*70)
print("INSPECT MODEL 1 (Jump-1Leg)")
print("="*70)

try:
    with open('model/sajojo_model1.pkl', 'rb') as f:
        model = joblib.load(f)
    
    print(f"\n🤖 MODEL INFO:")
    print(f"  - Type: {type(model).__name__}")
    print(f"  - Classes: {model.classes_}")
    print(f"  - Number of estimators: {model.n_estimators}")
    print(f"  - Max depth: {model.max_depth}")
    
    if hasattr(model, 'feature_importances_'):
        print(f"\n🎯 FEATURE IMPORTANCE (Top 10):")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        for i, idx in enumerate(indices):
            print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")
    
    # Test dengan data dummy
    print(f"\n🧪 TEST PREDICTION (using first row from training data):")
    
    # Get first Jump-1Leg sample
    df_jump = df[df[0] == 'Jump-1Leg']
    if len(df_jump) > 0:
        sample = df_jump.iloc[0, 1:].values.reshape(1, -1)
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        
        print(f"  - Actual: Jump-1Leg")
        print(f"  - Predicted: {pred}")
        print(f"  - Probabilities: {dict(zip(model.classes_, proba))}")
    
    # Get first Miss sample
    df_miss = df[df[0] == 'Miss']
    if len(df_miss) > 0:
        sample = df_miss.iloc[0, 1:].values.reshape(1, -1)
        pred = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0]
        
        print(f"\n  - Actual: Miss")
        print(f"  - Predicted: {pred}")
        print(f"  - Probabilities: {dict(zip(model.classes_, proba))}")
    
    # Check training set size
    if hasattr(model, 'n_samples_'):
        print(f"\n📊 TRAINING SET SIZE: {model.n_samples_}")
    
except FileNotFoundError:
    print("\n❌ Model file 'model/sajojo_model1.pkl' not found!")
    print("   Run retrain_sajojo_balanced.py first")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

# Recommendations
print("\n💡 RECOMMENDATIONS:")
print("\n1. Check if training data distribution is reasonable")
print("   - Each movement should have at least 100+ samples")
print("   - Ratio shouldn't be more than 1:3 (movement:Miss)")

print("\n2. If model still predicts 'Miss' for training samples:")
print("   - Model is severely overfitting or underfitting")
print("   - Need to check training process")

print("\n3. If feature importance is all zeros or very low:")
print("   - Features might not be discriminative enough")
print("   - Consider using different features or preprocessing")