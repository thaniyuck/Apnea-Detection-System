import numpy as np
import scipy.signal as signal
from scipy.ndimage import median_filter

def preprocess_ppg(raw_signal, fs=400):
    """
    Implements the paper's dual-stage signal purification pipeline.
    Removes baseline drift via median filtering and high-frequency noise via an IIR lowpass filter.
    """
    # ---------------------------------------------------------
    # STAGE 1: Baseline Drift Removal
    # ---------------------------------------------------------
    # The paper specifies a median filter with a sliding window length of 2.5N
    # (where N is the sampling frequency) to obtain the baseline drift signal.
    window_length = int(2.5 * fs)
    
    # The median filter function requires an odd window length mathematically
    if window_length % 2 == 0:
        window_length += 1  
        
    # Isolate the wandering baseline, then subtract it from the raw signal
    baseline_drift = median_filter(raw_signal, size=window_length)
    signal_no_drift = raw_signal - baseline_drift
    
    # ---------------------------------------------------------
    # STAGE 2: High-Frequency Noise Inhibition
    # ---------------------------------------------------------
    # The paper applies a 10-order IIR lowpass filter with a 4.8 Hz cutoff.
    nyquist = 0.5 * fs
    cutoff = 4.8
    normal_cutoff = cutoff / nyquist
    
    # Design the Butterworth filter coefficients (a type of IIR filter)
    b, a = signal.butter(10, normal_cutoff, btype='low', analog=False)
    
    # Use filtfilt for zero-phase filtering. This runs the filter forward and 
    # backward to completely eliminate phase delay artifacts, ensuring your 
    # peaks remain exactly where they occurred in time.
    clean_signal = signal.filtfilt(b, a, signal_no_drift)
    
    return clean_signal


def calculate_spo2(red_signal, ir_signal, fs=400):
    """
    Advanced Beat-by-Beat SpO2 Calculation.
    Uses rolling 1.5-second windows to extract true Peak-to-Trough AC amplitude 
    and dynamic DC baselines, guarded by a Perfusion Index (PI) threshold.
    """
    # 1.5 seconds guarantees we capture at least one full heartbeat (down to 40 BPM)
    window_size = int(1.5 * fs)  
    step_size = int(0.5 * fs)    # Overlap windows by 0.5 seconds for higher resolution
    
    valid_R_values = []
    
    # Iterate through the 60-second signal in 1.5-second chunks
    for i in range(0, len(red_signal) - window_size, step_size):
        red_window = red_signal[i : i + window_size]
        ir_window = ir_signal[i : i + window_size]
        
        # 1. Dynamic DC Extraction (Mean of the specific window)
        dc_red = np.mean(red_window)
        dc_ir = np.mean(ir_window)
        
        if dc_red == 0 or dc_ir == 0:
            continue
            
        # 2. True Peak-to-Trough AC Extraction
        # Subtract the DC to center the wave, then measure absolute height
        detrended_red = red_window - dc_red
        detrended_ir = ir_window - dc_ir
        
        ac_red = np.max(detrended_red) - np.min(detrended_red)
        ac_ir = np.max(detrended_ir) - np.min(detrended_ir)
        
        # 3. Perfusion Index (PI) Guard
        # If the blood flow is too weak (PI < 0.2%), the AC value is mostly noise.
        # Discard this window to prevent it from corrupting the final SpO2.
        pi_ir = (ac_ir / dc_ir) * 100
        if pi_ir < 0.2:
            continue
            
        # 4. Calculate Ratio of Ratios for this specific heartbeat
        R = (ac_red / dc_red) / (ac_ir / dc_ir)
        valid_R_values.append(R)

    # 5. Final Synthesis
    if len(valid_R_values) == 0:
        return 0 # Failsafe if the entire 60 seconds was pure noise or empty
        
    # Take the median of all valid R values to perfectly ignore any isolated extreme outliers
    median_R = np.median(valid_R_values)
    
    # Apply standard MAX301xx empirical calibration
    spo2 = 104.0 - (17.0 * median_R)
    
    # Clamp biological limits
    spo2 = max(0, min(100, int(spo2)))
    
    return spo2