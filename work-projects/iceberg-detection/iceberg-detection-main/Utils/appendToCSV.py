import csv
import os


def appendToCSV(file_path, preprocessing_method, model, score, f1_score_var):
    # Check if the file already exists
    file_exists = os.path.exists(file_path)

    # Open the file in append mode
    with open(file_path, mode="a", newline="") as csv_file:
        fieldnames = ["preprocessing_method", "model", "score", "score_var"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write header only if the file is being created for the first time
        if not file_exists:
            writer.writeheader()

        # Write the new data
        writer.writerow(
            {
                "preprocessing_method": preprocessing_method,
                "model": model,
                "score": score,
                "score_var": f1_score_var,
            }
        )
