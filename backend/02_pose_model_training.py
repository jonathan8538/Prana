from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

def get_data(file_path):
    """Load dan prepare data"""
    print(f"\n📂 Loading data dari: {file_path}")
    
    df = pd.read_csv(file_path, header=None)
    
    # X = features (pose landmarks), y = labels (nama gerakan)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    
    print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Distribusi kelas:")
    print(y.value_counts())
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val, y.unique()


def train_model(model, X_train, X_val, y_train, y_val, name=None, param_grid=None):
    """Training model dengan optional GridSearch"""
    
    print(f"\n{'='*60}")
    print(f"🔄 Training: {name}")
    print(f"{'='*60}")
    
    if param_grid:
        # Grid Search
        grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        
        print(f"\n✨ Best Parameters: {grid.best_params_}")
        print(f"✨ Best CV Score: {grid.best_score_:.4f}")
        
        best_model = grid.best_estimator_
        best_score = grid.best_score_
    else:
        # Direct training
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        print(f"\n📈 CV Scores: {cv_scores}")
        print(f"📈 Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        best_model = model
        best_score = cv_scores.mean()
    
    # Evaluasi di validation set
    y_pred = best_model.predict(X_val)
    
    print(f"\n📊 Validation Set Performance:")
    print(f"   Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"   Precision: {precision_score(y_val, y_pred, average='weighted'):.4f}")
    print(f"   Recall:    {recall_score(y_val, y_pred, average='weighted'):.4f}")
    print(f"   F1 Score:  {f1_score(y_val, y_pred, average='weighted'):.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_val, y_pred))
    
    return best_model, best_score


def find_best_model(X_train, X_val, y_train, y_val):
    """Coba beberapa model dan pilih yang terbaik"""
    
    models = [
        {
            'model': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            'params_grid': {
                'logisticregression__C': [0.01, 0.1, 1, 10, 100],
                'logisticregression__penalty': ['l2'],
                'logisticregression__solver': ['lbfgs', 'newton-cg']
            },
            'name': 'Logistic Regression'
        },
        {
            'model': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
            'params_grid': {
                'randomforestclassifier__n_estimators': [100, 200],
                'randomforestclassifier__max_depth': [5, 10, None],
                'randomforestclassifier__min_samples_split': [2, 5]
            },
            'name': 'Random Forest'
        },
        {
            'model': make_pipeline(StandardScaler(), KNeighborsClassifier()),
            'params_grid': {
                'kneighborsclassifier__n_neighbors': [3, 5, 7, 9],
                'kneighborsclassifier__weights': ['uniform', 'distance']
            },
            'name': 'K-Nearest Neighbors'
        }
    ]
    
    best_overall_model = None
    best_overall_score = -1
    best_overall_name = None
    
    for model_config in models:
        model, score = train_model(
            model_config['model'],
            X_train, X_val, y_train, y_val,
            name=model_config['name'],
            param_grid=model_config['params_grid']
        )
        
        if score > best_overall_score:
            best_overall_model = model
            best_overall_score = score
            best_overall_name = model_config['name']
    
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_overall_name}")
    print(f"🏆 BEST SCORE: {best_overall_score:.4f}")
    print(f"{'='*60}")
    
    return best_overall_model, best_overall_name, best_overall_score


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-data", type=str, 
                    default='tari_bali_training.csv',
                    help="Nama file CSV training data")
    ap.add_argument("--model-name", type=str, 
                    default='tari_bali_model',
                    help="Nama model yang akan disimpan (tanpa .pkl)")
    
    args = vars(ap.parse_args())
    
    model_name = args['model_name']
    training_data_path = f"data/{args['training_data']}"
    
    # Load data
    X_train, X_val, y_train, y_val, classes = get_data(training_data_path)
    
    # Find best model
    best_model, best_name, best_score = find_best_model(
        X_train, X_val, y_train, y_val
    )
    
    # Simpan model
    import os
    os.makedirs('model', exist_ok=True)
    
    model_path = f"model/{model_name}.pkl"
    joblib.dump(best_model, model_path)
    
    # Simpan classes
    with open(f'model/{model_name}_classes.txt', 'w') as f:
        f.write(','.join(classes))
    
    # Simpan metadata
    with open(f'model/{model_name}_metadata.txt', 'w') as f:
        f.write(f"Model: {best_name}\n")
        f.write(f"Score: {best_score:.4f}\n")
        f.write(f"Classes: {', '.join(classes)}\n")
    
    print(f"\n✅ Model disimpan ke: {model_path}")
    print(f"✅ Classes disimpan ke: model/{model_name}_classes.txt")
    print(f"✅ Metadata disimpan ke: model/{model_name}_metadata.txt")