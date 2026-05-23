import os
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import your clean math modules
from src.signal_processing import preprocess_ppg
from src.feature_extraction import extract_all_features

def generate_clinical_dataset():
    data_dir = "./data/raw_mit_bih"
    csv_path = "./data/mit_bih_processed_features.csv"
    
    # Downloading a mix of severe, mild, and healthy patient records to balance the dataset
    records_to_download = ["slp01a", "slp01b", "slp02a", "slp02b", "slp03", "slp04"]
    
    os.makedirs(data_dir, exist_ok=True)
    dataset_rows = []
    
    for record_name in records_to_download:
        if not os.path.exists(os.path.join(data_dir, f"{record_name}.dat")):
            print(f"Downloading clinical record '{record_name}'...")
            wfdb.dl_database('slpdb', dl_dir=data_dir, records=[record_name])
        else:
            print(f"Record '{record_name}' found locally.")

        record_path = os.path.join(data_dir, record_name)
        record = wfdb.rdrecord(record_path)
        annotations = wfdb.rdann(record_path, 'st')
        
        # Isolate the PPG channel
        ppg_idx = record.sig_name.index('Pleth') if 'Pleth' in record.sig_name else 0
        signal_data = record.p_signal[:, ppg_idx]
        fs = record.fs
        
        samples_per_window = int(60 * fs)
        num_windows = len(signal_data) // samples_per_window
        
        print(f"Slicing {num_windows} minutes of data from {record_name}...")
        
        for i in tqdm(range(num_windows)):
            start_idx = i * samples_per_window
            end_idx = start_idx + samples_per_window
            raw_window = signal_data[start_idx:end_idx]
            
            # Check medical labels for this 60s window
            window_ann_indices = np.where((annotations.sample >= start_idx) & (annotations.sample < end_idx))[0]
            is_apnea = 0
            for idx in window_ann_indices:
                if 'A' in str(annotations.aux_note[idx]).upper() or 'APNEA' in str(annotations.aux_note[idx]).upper():
                    is_apnea = 1
                    break
                    
            try:
                # Pass data through YOUR specific modules
                clean_window = preprocess_ppg(raw_window, fs)
                features = extract_all_features(clean_window, fs)
                
                if features:
                    prv_dict, mse_list = features
                    row = {**prv_dict}
                    for j, val in enumerate(mse_list):
                        row[f'd_{j+1}'] = val
                    
                    # Target label for the model to predict
                    row['Label'] = is_apnea
                    
                    # Crucial for Leave-One-Subject-Out (LOSO) cross-validation
                    row['Subject'] = record_name 
                    
                    dataset_rows.append(row)
            except:
                pass # Skip corrupted chunks
                
    # Save the final CSV
    df = pd.DataFrame(dataset_rows)
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ Balanced dataset successfully generated at: {csv_path}")
    print(f"Total Patterns Extracted: {len(df)}")
    print("\nClass Balance:")
    print(df['Label'].value_counts())

if __name__ == "__main__":
    generate_clinical_dataset()