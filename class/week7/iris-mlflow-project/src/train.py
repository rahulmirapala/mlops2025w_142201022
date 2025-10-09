#src/train.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_explore_data():
    """Load and explore the Iris dataset"""
    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    df = X.copy()
    df['species'] = y
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Classes: {iris.target_names}")
    print(f"Class distribution:\n{df['species_name'].value_counts()}")
    return X, y, df

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Split and preprocess the data"""
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }

def train_random_forest(X_train, y_train, X_test, y_test, **params):
    """Train Random Forest model and log to an ACTIVE MLflow run"""
    mlflow.log_params(params)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    for metric, value in train_metrics.items():
        mlflow.log_metric(f"train_{metric}", value)
    for metric, value in test_metrics.items():
        mlflow.log_metric(f"test_{metric}", value)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix Random Forest')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix_rf.png')
    mlflow.log_artifact('confusion_matrix_rf.png')
    plt.close()
    signature = infer_signature(X_test, y_pred_test)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_test.head(3)
    )
    mlflow.set_tag("model_family", "tree_based")
    mlflow.set_tag("dataset", "iris")
    print(f"Random Forest Test Accuracy: {test_metrics['accuracy']:.4f}")
    return model, test_metrics['accuracy']

def train_logistic_regression(X_train, y_train, X_test, y_test, **params):
    """Train Logistic Regression model and log to an ACTIVE MLflow run"""
    mlflow.log_params(params)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    for metric, value in train_metrics.items():
        mlflow.log_metric(f"train_{metric}", value)
    for metric, value in test_metrics.items():
        mlflow.log_metric(f"test_{metric}", value)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix Logistic Regression')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix_lr.png')
    mlflow.log_artifact('confusion_matrix_lr.png')
    plt.close()
    signature = infer_signature(X_test, y_pred_test)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_test.head(3)
    )
    mlflow.set_tag("model_family", "linear")
    mlflow.set_tag("dataset", "iris")
    print(f"Logistic Regression Test Accuracy: {test_metrics['accuracy']:.4f}")
    return model, test_metrics['accuracy']

def hyperparameter_tuning():
    """Perform hyperparameter tuning with nested runs"""
    X, y, df = load_and_explore_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    mlflow.set_experiment("Iris Classification Hyperparameter Tuning")
    with mlflow.start_run(run_name="Hyperparameter Tuning Session") as parent_run:
        best_accuracy = 0
        best_model = None
        best_run_id = None
        rf_params = [
            {'n_estimators': 50, 'max_depth': 3, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 7, 'random_state': 42},
            {'n_estimators': 200, 'max_depth': None, 'random_state': 42}
        ]
        for params in rf_params:
            with mlflow.start_run(nested=True, run_name=f"RF_estimators_{params['n_estimators']}_depth_{params['max_depth']}") as child_run:
                model, accuracy = train_random_forest(X_train, y_train, X_test, y_test, **params)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_run_id = mlflow.active_run().info.run_id
        lr_params = [
            {'C': 0.1, 'random_state': 42, 'max_iter': 1000},
            {'C': 1.0, 'random_state': 42, 'max_iter': 1000},
            {'C': 10.0, 'random_state': 42, 'max_iter': 1000},
        ]
        for params in lr_params:
            with mlflow.start_run(nested=True, run_name=f"LR_C_{params['C']}"):
                model, accuracy = train_logistic_regression(X_train, y_train, X_test, y_test, **params)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_run_id = mlflow.active_run().info.run_id
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.log_param("best_run_id", best_run_id)
        mlflow.set_tag("status", "completed")
        print(f"\nBest model accuracy: {best_accuracy:.4f}")
        print(f"Best run ID: {best_run_id}")
        return best_model, best_run_id

def model_registry_example(best_run_id):
    """Demonstrate model registry functionality"""
    model_name = "iris-classifier"
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered: {model_name}, Version: {model_version.version}")
    return model_name, model_version.version

def load_and_predict():
    """Load model from registry and make predictions"""
    model_name = "iris-classifier"
    try:
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production", "Archived"])
        if not latest_versions:
            print(f"No versions found for model '{model_name}'. Skipping prediction.")
            return None
        latest_version_num = latest_versions[0].version
        model_uri = f"models:/{model_name}/{latest_version_num}"
        print(f"Loading model from URI: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    X, y, df = load_and_explore_data()
    X_sample = X.head(5)
    predictions = model.predict(X_sample)
    results = pd.DataFrame({
        'sepal_length': X_sample['sepal length (cm)'],
        'sepal_width': X_sample['sepal width (cm)'],
        'petal_length': X_sample['petal length (cm)'],
        'petal_width': X_sample['petal width (cm)'],
        'predicted_class': predictions,
        'predicted_species': [load_iris().target_names[p] for p in predictions]
    })
    print("\nPrediction Results:")
    print(results.to_string(index=False))
    return results

def main():
    """Main function to run the complete pipeline"""
    print("=== MLflow Iris Classification Pipeline ===\n")
    best_model, best_run_id = hyperparameter_tuning()
    model_name, version_info = model_registry_example(best_run_id)
    # THIS IS THE FIX: The variable 'version_info' already holds the version number.
    version = version_info if version_info else "N/A"
    results = load_and_predict()
    print(f"\n=== Pipeline Completed Successfully ===")
    print(f"Best model registered as: {model_name} (Version: {version})")
    print(f"MLflow UI: http://127.0.0.1:5000")
    # Clean up artifact files
    if os.path.exists('confusion_matrix_rf.png'):
        os.remove('confusion_matrix_rf.png')
    if os.path.exists('confusion_matrix_lr.png'):
        os.remove('confusion_matrix_lr.png')

if __name__ == "__main__":
    main()