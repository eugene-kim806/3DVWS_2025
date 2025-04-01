from collections import defaultdict, Counter
from pathlib import Path
import csv
import pandas as pd
import numpy as np

def get_patient_timepoints(args):
    """
    Extracts patients and their timepoints from dataset structure.
    """
    patient_timepoints = defaultdict(int)
    timepoint_paths = {}

    for timepoint_dir in sorted(Path(args.trainval_dir).iterdir()):
        assert timepoint_dir.is_dir(), f"!!! NOT A DIRECTORY: {timepoint_dir}"
        patient_ID = timepoint_dir.name[:-4]  # e.g., BraTS-GLI-00005
        timepoint_ID = timepoint_dir.name  # e.g., BraTS-GLI-00005-100

        # Store paths for different modalities
        modality_paths = {}
        for file in timepoint_dir.iterdir():
            modality = file.name[20:-7]  # t1c, t1n, t2f, t2w, or seg
            assert modality in ["t1c", "t1n", "t2f", "t2w", "seg"], f"!!! INVALID MODALITY NAME: {modality}"
            modality_paths[modality] = str(file)
        patient_timepoints[patient_ID] += 1
        timepoint_paths[timepoint_ID] = modality_paths

    return patient_timepoints, timepoint_paths


def split_patients(patient_timepoints, num_folds:int):
    """
    Distributes patients into balanced folds based on timepoints.
    """
    # Sort the list of (patient ID, number of timepoints) based on the number of timepoints in descending order
    patients = sorted(patient_timepoints.items(), key=lambda x: x[1], reverse=True)
    folds = [[] for _ in range(num_folds)]  # e.g., [[fold 1], [fold 2], [fold 3], [fold 4], [fold 5]]
    fold_timepoints = [0] * num_folds  # e.g., [0, 0, 0, 0, 0]

    for patient_ID, timepoints in patients:  # e.g., ('BraTS-GLI-00005', 2)
        min_fold_idx = fold_timepoints.index(min(fold_timepoints))  # index of the fold with minimum timepoints
        folds[min_fold_idx].append(patient_ID)
        fold_timepoints[min_fold_idx] += timepoints

    return folds, fold_timepoints  # fold_timepoints = [325, 324, 324, 324, 324]


def generate_splits(args):
    """
    Creates CSVs with different training-validation splits.
    """
    split_save_dir = Path(args.split_folds_dir)
    split_save_dir.mkdir(parents=True, exist_ok=True)

    patient_timepoints, timepoint_paths = get_patient_timepoints(args)
    folds, fold_timepoints = split_patients(patient_timepoints, args.num_folds)

    if args.verbose:
        split_logs_dir = split_save_dir / 'logs'
        split_logs_dir.mkdir(parents=True, exist_ok=True)
        split_log_file = split_logs_dir / 'split_distribution_log.txt'

        total_patients = len(patient_timepoints)
        total_timepoints = sum(patient_timepoints.values())
        log_texts = []
        log_texts += ["**Fold Distribution Summary**", 
                      f"  - Total Patients: {total_patients}",
                      f"  - Total Timepoints: {total_timepoints}"]
    
    for valid_fold_idx in range(args.num_folds):
        split_csv_path = split_save_dir / f"split_fold_{valid_fold_idx + 1}.csv"

        with open(split_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["patient_ID", "timepoint_ID", "split", "t1n", "t1c", "t2f", "t2w", "seg"]
            writer.writerow(header)

            for fold_idx, fold in enumerate(folds):
                split_label = "valid" if fold_idx == valid_fold_idx else "train"
                for patient_id in fold:
                    for timepoint_id, paths in timepoint_paths.items():
                        if patient_id in timepoint_id:
                            writer.writerow([patient_id, timepoint_id, split_label] + [paths.get(modality, "") for modality in header[3:]])
            
            if args.verbose:
                log_texts += [f"========================= Fold {valid_fold_idx + 1} =========================",
                            f"➝ CSV saved at: {split_csv_path}",
                            f"# Validation: {len(folds[valid_fold_idx])} Patients, {fold_timepoints[valid_fold_idx]} Total Timepoints"]
                for timepoint_count, num_patients in sorted(Counter(patient_timepoints[pID] for pID in folds[valid_fold_idx]).items(), reverse=True):
                    log_texts.append(f"   - {num_patients} Patients with {timepoint_count} Timepoint{'s' if timepoint_count > 1 else ''}")
    
    if args.verbose:
        with open(split_log_file, "w") as log_file:
            log_file.write("\n".join(log_texts) + "\n")



def load_cases_split(split_path):
    df = pd.read_csv(split_path)
    cases_name, cases_split = np.array(df['timepoint_ID']), np.array(df['split'])
    train_cases = list(cases_name[cases_split == 'train'])
    valid_cases = list(cases_name[cases_split == 'valid'])

    return train_cases, valid_cases
