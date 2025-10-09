import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
def compare_experiments():
    """Compare all experiments and runs"""

    client = MlflowClient()

    # Get experiment by name
    experiment = client.get_experiment_by_name("Iris Classification Hyperparameter Tuning")

    if experiment:
        # Get all runs from the experiment
        runs = client.search_runs(experiment.experiment_id)

        # Create comparison DataFrame
        comparison_data = []
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                'model_type': run.data.params.get('model_type', 'Unknown'),
                'test_accuracy': run.data.metrics.get('test_accuracy', 0),
                'test_f1_score': run.data.metrics.get('test_f1_score', 0),
                'status': run.info.status
            }
            comparison_data.append(run_data)

        # Sort by test accuracy
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('test_accuracy', ascending=False)

        print("Model Comparison Results:")
        print("=" * 80)
        print(df.to_string(index=False))

        return df
    else:
        print("Experiment not found!")
        return None

if __name__ == "__main__":
    compare_experiments()
