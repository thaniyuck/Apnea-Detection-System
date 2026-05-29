from src.data_acquisition import PPGDataCollector
from src.signal_processing import preprocess_ppg, calculate_spo2
from src.feature_extraction import extract_all_features
from src.ml_model import ApneaClassifier
import numpy as np

# --- GLOBALS ---
FS = 400
hardware = None # Global reference for hardware access

def process_live_window(red_np, ir_np):
    """
    This callback is triggered automatically by data_acquisition.py 
    every time the 60-second hardware buffer fills up.
    """
    global hardware
    print("\n" + "="*40)
    print(" LIVE DIAGNOSTIC UPDATE ")
    print("="*40)

    # --- THE PROXIMITY GUARD ---
    mean_ir = np.mean(ir_np)
    
    # DEBUG: Menampilkan nilai asli sensor
    print(f" [Debug] Sensor IR Baseline: {mean_ir:.0f}")

    # ==============================================================
    # CALIBRATION TUNED: Threshold dinaikkan menjadi 120000 
    # (Titik tengah antara 90k saat kosong dan 190k saat ada jari)
    # ==============================================================
    if mean_ir < 120000:  
        # Send Error 'E' to ESP32 OLED
        if hardware:
            try:
                hardware.ser.write("0,0,E\n".encode('utf-8'))
            except Exception as e:
                print(f" [!] Failed to update OLED: {e}")
        
        print(" Status:               📭 NO FINGER DETECTED")
        print(" Action:               Please place finger on sensor to resume.")
        return 

    # 1. Calculate SpO2
    spo2 = calculate_spo2(red_np, ir_np)
    
    # 2. Clean BOTH signals and extract features properly
    clean_red = preprocess_ppg(red_np, FS)
    clean_ir = preprocess_ppg(ir_np, FS)
    features = extract_all_features(clean_red, clean_ir, FS)
    
    print(f" Blood Oxygen (SpO2): {int(spo2)}%")
    
    if features:
        prv_dict, mse_list = features
        
        hr = prv_dict.get('PR (BPM)', 0)
        print(f" Heart Rate:          {hr:.2f} BPM")
        
        # 3. Run Inference
        prediction = diagnostic_engine.predict_live_window(prv_dict, mse_list)
        status = "🚨 APNEA DETECTED" if prediction == 1 else "💚 NORMAL"
        print(f" Status:              {status}")
        
        # --- 4. SEND DATA TO OLED ---
        status_char = "A" if prediction == 1 else "N"
        lcd_message = f"{int(hr)},{int(spo2)},{status_char}\n"
        
        if hardware:
            try:
                hardware.ser.write(lcd_message.encode('utf-8'))
            except Exception as e:
                print(f" [!] Failed to update OLED: {e}")
        
        # 5. Expose the Math
        sdnn = prv_dict.get('SDNN_PP (ms)', 0)
        d1 = mse_list[0] if len(mse_list) > 0 else 0
        
        if prediction == 1:
            print(f" Reasoning:           Autonomic stress / Complexity loss.")
        else:
            print(f" Reasoning:           Stable autonomic tone / Normal complexity.")
            
        print(f" -> Live PRV (SDNN_PP): {sdnn:.4f}")
        print(f" -> Live ME (Scale 1):  {d1:.4f}")
            
    else:
        print(" [!] Signal unstable. Adjust ring and hold still.")


if __name__ == "__main__":
    print("Booting SAS Diagnostic System...")
    
    # 1. Initialize Machine Learning Core
    diagnostic_engine = ApneaClassifier()
    
    # Check if we need to train the model, or just load it
    if not diagnostic_engine.load_model():
        print("Training model from scratch...")
        success = diagnostic_engine.train_model(csv_path="data/mit_bih_processed_features.csv")
        if not success:
            print("CRITICAL: Run 'python build_dataset.py' first.")
            exit()
            
    # 2. Initialize Hardware and Start Stream
    hardware = PPGDataCollector(port='COM12', baud_rate=115200, fs=FS)
    
    # Pass our processing function into the hardware listener
    hardware.start_streaming(processing_callback=process_live_window)