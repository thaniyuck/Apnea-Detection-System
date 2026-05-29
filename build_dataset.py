import os
import time
import requests
import wfdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.signal as signal

# Import your clean math modules
from src.signal_processing import preprocess_ppg
from src.feature_extraction import calculate_prv_features, calculate_mse_features

def download_file_with_resume(url, dest_path, max_retries=15):
    """
    Bulletproof HTTP downloader. Resumes partial downloads exactly where they dropped.
    """
    for attempt in range(max_retries):
        try:
            headers = {}
            file_mode = 'ab' # Append mode by default

            if os.path.exists(dest_path):
                downloaded_bytes = os.path.getsize(dest_path)
                headers['Range'] = f'bytes={downloaded_bytes}-'
            else:
                downloaded_bytes = 0
                file_mode = 'wb' # Write mode if it doesn't exist

            response = requests.get(url, headers=headers, stream=True, timeout=15)

            if response.status_code == 416: 
                # HTTP 416: Range Not Satisfiable (The file is already 100% downloaded!)
                return True
            elif response.status_code == 200: 
                # Server ignored Range, starting over
                file_mode = 'wb'
            elif response.status_code not in [200, 206]:
                print(f"      [X] Server returned HTTP {response.status_code}")
                return False

            with open(dest_path, file_mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            current_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
            print(f"      [!] Connection dropped. Resuming from byte {current_size:,}... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(2)
    return False

def fetch_apnea_record(record_name, dl_dir, is_resp=False):
    """
    Determines exactly which files (.hea, .dat, .apn) to fetch from PhysioNet.
    """
    base_url = "https://physionet.org/files/apnea-ecg/1.0.0/"
    extensions = ['.hea', '.dat']
    
    if not is_resp:
        extensions.append('.apn') # Only base records have doctor annotations

    for ext in extensions:
        file_name = f"{record_name}{ext}"
        url = f"{base_url}{file_name}"
        dest_path = os.path.join(dl_dir, file_name)
        
        print(f"      -> Verifying {file_name}...")
        download_file_with_resume(url, dest_path)

def generate_clinical_dataset():
    data_dir = "./data/raw_apnea_ecg"
    csv_path = "./data/apnea_ecg_processed_features.csv"
    
    # Base names for the 8 patients with full sensor suites
    base_records = ['a01', 'a02', 'a03', 'a04', 'b01', 'c01', 'c02', 'c03']
    
    os.makedirs(data_dir, exist_ok=True)
    dataset_rows = []
    
    for base_name in base_records:
        resp_name = f"{base_name}r"
        
        print(f"\n--- Checking Record: {base_name} ---")
        # 1. Download Base Record (ECG + Labels) using Bulletproof Downloader
        fetch_apnea_record(base_name, data_dir, is_resp=False)
                
        # 2. Download Resp Record (SpO2) using Bulletproof Downloader
        fetch_apnea_record(resp_name, data_dir, is_resp=True)
                
        try:
            # 3. Load the parallel tracks simultaneously using wfdb
            ecg_record = wfdb.rdrecord(os.path.join(data_dir, base_name))
            spo2_record = wfdb.rdrecord(os.path.join(data_dir, resp_name))
            annotations = wfdb.rdann(os.path.join(data_dir, base_name), 'apn')
        except Exception as e:
            print(f"\n[!] Failed to read {base_name} into memory: {e}. Skipping.")
            continue
            
        # 4. Isolate ECG
        ecg_idx = 0
        signal_data = ecg_record.p_signal[:, ecg_idx]
        
        # 5. Isolate SpO2 from the 'r' file
        sao2_idx = -1
        for i, name in enumerate(spo2_record.sig_name):
            if 'SAO2' in name.upper() or 'SPO2' in name.upper():
                sao2_idx = i
                break
                
        if sao2_idx == -1:
            print(f"Skipping {base_name}: No clinical oxygen track found in {resp_name}.")
            continue
            
        sao2_data = spo2_record.p_signal[:, sao2_idx]
        
        fs = ecg_record.fs
        samples_per_window = int(60 * fs)
        
        # Ensure we only iterate as far as the shortest file allows
        num_windows = min(len(signal_data), len(sao2_data)) // samples_per_window
        
        print(f"\nSlicing {num_windows} minutes of synchronized data from {base_name}...")
        
        for i in tqdm(range(num_windows)):
            start_idx = i * samples_per_window
            end_idx = start_idx + samples_per_window
            
            raw_window = signal_data[start_idx:end_idx]
            sao2_window = sao2_data[start_idx:end_idx]
            
            # 6. Check medical labels for this 60s window
            window_ann_indices = np.where((annotations.sample >= start_idx) & (annotations.sample < end_idx))[0]
            is_apnea = 0
            for idx in window_ann_indices:
                symbol = str(annotations.symbol[idx]).upper() if hasattr(annotations, 'symbol') else ""
                note = str(annotations.aux_note[idx]).upper() if hasattr(annotations, 'aux_note') else ""
                
                if 'A' in symbol or 'APNEA' in note:
                    is_apnea = 1
                    break
                    
            try:
                # 7. Extract PRV (Timing & Amplitude)
                clean_window = preprocess_ppg(raw_window, fs)
                prv_features = calculate_prv_features(clean_window, fs)
                
                if prv_features:
                    # 8. Extract SpO2 (Oxygen)
                    clean_sao2 = sao2_window[~np.isnan(sao2_window)]
                    if len(clean_sao2) == 0:
                        continue # Skip window if SpO2 is missing
                        
                    window_spo2 = np.median(clean_sao2)
                    prv_features["SpO2 (%)"] = round(window_spo2, 2)
                    
                    # 9. Extract MSE (Entropy/Complexity)
                    if fs > 100:
                        target_len = int(len(clean_window) * (100 / fs))
                        mse_signal = signal.resample(clean_window, target_len)
                    else:
                        mse_signal = clean_window
                        
                    mse_list = calculate_mse_features(mse_signal)
                    
                    # Assemble the row
                    row = {**prv_features}
                    for j, val in enumerate(mse_list):
                        row[f'd_{j+1}'] = val
                    
                    row['Label'] = is_apnea
                    row['Subject'] = base_name 
                    
                    dataset_rows.append(row)
            except Exception:
                pass # Skip corrupted chunks
                
    # 10. Save the final CSV with Temporal Tracking
    df = pd.DataFrame(dataset_rows)
    if len(df) > 0:
        print("\nCalculating temporal features (Delta SpO2)...")
        df = df.sort_values(by=['Subject'])
        df['SpO2_Delta'] = df.groupby('Subject')['SpO2 (%)'].diff().fillna(0.0)
        
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Balanced 3-Parameter Dataset successfully generated at: {csv_path}")
        print(f"Total Patterns Extracted: {len(df)}")
        print("\nClass Balance:")
        print(df['Label'].value_counts())
    else:
        print("\n❌ No valid data extracted. Check your raw files.")

if __name__ == "__main__":
    generate_clinical_dataset()