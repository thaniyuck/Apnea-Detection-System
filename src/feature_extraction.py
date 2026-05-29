import numpy as np
import scipy.signal as signal
import antropy as ant

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
        if dc_ir == 0:
            continue
        pi_ir = (ac_ir / dc_ir) * 100
        if pi_ir < 0.2:
            continue
            
        # 4. Calculate Ratio of Ratios for this specific heartbeat
        R = (ac_red / dc_red) / (ac_ir / dc_ir)
        valid_R_values.append(R)

    # 5. Final Synthesis
    if len(valid_R_values) == 0:
        return 0 # Failsafe if the entire window was pure noise or empty
        
    # Take the median of all valid R values to perfectly ignore isolated extreme outliers
    median_R = np.median(valid_R_values)
    
    # ==============================================================
    # CALIBRATION TUNED: Shifted baseline from 106.5 to 108.5 to push SpO2 up by +2%
    # ==============================================================
    spo2 = 108.5 - (17.0 * median_R)
    
    # Clamp biological limits
    spo2 = max(0, min(100, int(spo2)))
    
    return spo2

def calculate_prv_features(stable_signal, fs=400):
    """
    Extracts Time-Domain, Amplitude, and Frequency-Domain (LF/HF) parameters.
    Uses a strict lockout window and dynamic prominence for peak detection.
    """
    # Peak Detection lockout reduced to 300ms (0.30s) and prominence added to catch missed beats
    lockout_samples = int(0.30 * fs) 
    dynamic_prominence = np.std(stable_signal) * 0.4
    peaks, _ = signal.find_peaks(stable_signal, distance=lockout_samples, prominence=dynamic_prominence)
    
    # Calculate peak-to-peak intervals (PPI) in milliseconds
    ppi = np.diff(peaks) * (1000.0 / fs)
    
    # Return None if the signal window lacks enough clear beats for valid math
    if len(ppi) < 5:
        return None
        
    # 2. Extract Time-Domain Features
    pr = 60000.0 / np.mean(ppi)        # PR: Pulse rate (beats per minute)
    sdnn = np.std(ppi)                 # SDNN_PP: Standard deviation of peak-to-peak periods
    pp_median = np.median(ppi)         # PP_median: Median periods of peak-to-peak
    
    ppi_diff = np.abs(np.diff(ppi))
    nn50 = np.sum(ppi_diff > 50)       # NN50: Adjacent peak intervals differing by > 50ms
    pnn50 = (nn50 / len(ppi)) * 100    # PNN50: Percentage of NN50
    
    # 3. Extract Amplitude Features
    amplitudes = stable_signal[peaks]
    aa = np.mean(amplitudes)           # AA: Average peak-to-peak amplitude
    aa_median = np.median(amplitudes)  # AA_median: Median peak-to-peak amplitude
    sdnn_aa = np.std(amplitudes)       # SDNN_AA: Standard deviation of amplitude
    
    # 4. Extract Frequency-Domain Features (LF/HF Ratio via Lomb-Scargle)
    try:
        # Get the actual time of each interval in seconds
        t_beats = peaks[1:] / fs
        
        # Mean-center the intervals to remove the DC (0 Hz) component
        ppi_centered = ppi - np.mean(ppi)
        
        # Define the biological frequency bands in Hz
        f_lf = np.linspace(0.04, 0.15, 500)
        f_hf = np.linspace(0.15, 0.40, 500)
        
        # Convert to angular frequencies (radians/sec) for Scipy's lombscargle
        w_lf = 2 * np.pi * f_lf
        w_hf = 2 * np.pi * f_hf
        
        # Compute the spectral power using Lomb-Scargle for unevenly sampled data
        pgram_lf = signal.lombscargle(t_beats, ppi_centered, w_lf, normalize=True)
        pgram_hf = signal.lombscargle(t_beats, ppi_centered, w_hf, normalize=True)
        
        # Integrate the area under the curve to get total power for each band
        lf_power = np.trapz(pgram_lf, f_lf)
        hf_power = np.trapz(pgram_hf, f_hf)
        
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0
    except Exception:
        lf_hf_ratio = 0.0
    
    # CRITICAL: These exact keys match the Random Forest training dataset features
    return {
        "PR (BPM)": round(pr, 2),
        "SDNN_PP (ms)": round(sdnn, 2),
        "PP_median (ms)": round(pp_median, 2),
        "NN50 (counts)": int(nn50),
        "PNN50 (%)": round(pnn50, 2),
        "AA": round(aa, 2),
        "AA_median": round(aa_median, 2),
        "SDNN_AA": round(sdnn_aa, 2),
        "LF/HF Ratio": round(lf_hf_ratio, 3)
    }

def improved_coarse_graining(sequence, tau):
    """
    Implements the 50% sliding window coarse-graining algorithm.
    """
    if tau == 1:
        return sequence
    step = max(1, tau // 2)
    coarse_sequence = []
    for i in range(0, len(sequence) - tau + 1, step):
        coarse_sequence.append(np.mean(sequence[i:i + tau]))
    return np.array(coarse_sequence)

def calculate_mse_features(stable_signal, max_scales=15):
    """
    Calculates sample entropy across multiple scales to measure signal complexity.
    """
    mse_list = []
    m = 2                              # Embedding dimension
    r = 0.15 * np.std(stable_signal)  # Noise tolerance threshold
    
    for tau in range(1, max_scales + 1):
        coarse_data = improved_coarse_graining(stable_signal, tau)
        try:
            samp_en = ant.sample_entropy(coarse_data, order=m, metric='chebyshev')
            if np.isnan(samp_en) or np.isinf(samp_en):
                samp_en = 0.0
        except Exception:
            samp_en = 0.0
        mse_list.append(round(samp_en, 4))
    return mse_list

def extract_all_features(filtered_red, filtered_ir, fs=400):
    """
    Master function that strips edge artifacts, downsamples for ML compatibility,
    and extracts all parameters required for the Random Forest classifier.
    """
    # Clip the last 3 seconds to bypass IIR filter boundary transients
    stable_red = filtered_red[:-int(3 * fs)]
    stable_ir = filtered_ir[:-int(3 * fs)]
    
    # 1. Extract SpO2 using the robust beat-by-beat calculator
    spo2_val = calculate_spo2(stable_red, stable_ir, fs)
    
    # 2. Extract PRV at full 400Hz resolution (Using the IR channel as it penetrates deeper)
    prv_features = calculate_prv_features(stable_ir, fs)
    if not prv_features:
        return None
        
    # Inject SpO2 into the dictionary so it becomes a column in your dataset
    prv_features["SpO2 (%)"] = spo2_val
        
    # 3. Downsample to 100Hz for Entropy features (to match ML training logic)
    if fs > 100:
        target_len = int(len(stable_ir) * (100 / fs))
        mse_signal = signal.resample(stable_ir, target_len)
    else:
        mse_signal = stable_ir
        
    mse_features = calculate_mse_features(mse_signal)
    
    return prv_features, mse_features