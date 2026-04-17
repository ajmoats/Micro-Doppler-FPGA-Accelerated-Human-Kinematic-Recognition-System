import os
import scipy.io as sio
import numpy as np
import pickle

def extract_person_info(filename):
    """
    Extract person ID and run number from filename.
    Example:
        dm.mat   -> (dm, 1)
        dm_2.mat -> (dm, 2)
    """
    name = filename.replace(".mat", "")
    
    if "_" in name:
        person, run = name.split("_")
        run = int(run)
    else:
        person = name
        run = 1
    
    return person, run


def process_data():
    data_path = "../data"
    all_data = []

    for filename in os.listdir(data_path):
        if filename.endswith(".mat"):
            print(f"Processing {filename}...")

            filepath = os.path.join(data_path, filename)
            mat_data = sio.loadmat(filepath)

            # Extract person info
            person_id, run_id = extract_person_info(filename)

            # Extract action labels (MATLAB cell array → Python list)
            if "lblMaster" in mat_data:
                lbl_master = [str(x[0]) for x in mat_data["lblMaster"].squeeze()]
            else:
                print(f"Warning: lblMaster not found in {filename}")
                continue

            # Keep only first 21 actions
            lbl_master = lbl_master[:21]

            # Store structured info
            file_record = {
                "filename": filename,
                "person_id": person_id,
                "run_id": run_id,
                "action_labels": lbl_master,
                "raw_data": mat_data  # optional (can remove if too large)
            }

            all_data.append(file_record)

    # Save processed dataset
    output_path = os.path.join(data_path, "processed_data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_data, f)

    print(f"\nSaved processed data to {output_path}")


if __name__ == "__main__":
    process_data()