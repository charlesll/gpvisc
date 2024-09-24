import os
import pandas as pd

def get_metrics(folder_path):
    """Read the metrics.csv file from a given folder and return the test RMSE."""
    metrics_file = os.path.join(folder_path, "metrics.csv")
    if os.path.exists(metrics_file):
        metrics = pd.read_csv(metrics_file)
        # Assuming the first row has the relevant RMSE values
        # handtest is the test dataset without the SciGlass data
        return metrics["test"].iloc[0], metrics["handtest"].iloc[0]  
    else:
        raise FileNotFoundError(f"metrics.csv not found in {folder_path}")

def main(base_path, num_models=3):
    """Select the top `num_models` best models based on test RMSE."""
    folders = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    results = []

    for folder in folders:
        try:
            test_rmse, handtest_rmse = get_metrics(folder)
            results.append((folder, test_rmse, handtest_rmse))
        except Exception as e:
            print(f"Error processing {folder}: {e}")

    # Sort the results by test RMSE (ascending order)
    sorted_results = sorted(results, key=lambda x: x[1])

    # Select the top `num_models` folders
    best_models = sorted_results[:num_models]

    print(f"Top {num_models} models:")
    for i, (folder, rmse, rmse2) in enumerate(best_models, 1):
        print(f"{i}. Folder: {folder}, Test RMSE: {rmse}, , Test RMSE (no sciglass): {rmse2}")

    return best_models

if __name__ == "__main__":
    base_path = "./models/GP/"  # directory containing the model folders
    best_models = main(base_path, num_models=6)
