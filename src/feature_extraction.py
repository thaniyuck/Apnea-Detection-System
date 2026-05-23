import numpy as np
import scipy.signal as signal
import antropy as ant

def calculate_prv_features(stable_signal, fs=400):
    """
    Extracts the 8 Pulse Rate Variability (PRV) parameters.
    Uses a strict 400ms anti-false detection lockout window to bypass the dicrotic notch.
    """
    # 1. Peak Detection with 400ms lockout (0.40s * fs)
    lockout_samples = int(0.40 * fs) 
    peaks, _ = signal.find_peaks(stable_signal, distance=lockout_samples)
    
    # Calculate peak-to-peak intervals (PPI) in milliseconds
    ppi = np.diff(peaks) * (1000.0 / fs)
    
    # Return None if the signal window lacks enough clear beats for a valid mathematical calculation
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
    
    # CRITICAL: These exact keys match the Random Forest training dataset features
    return {
        "PR (BPM)": round(pr, 2),
        "SDNN_PP (ms)": round(sdnn, 2),
        "PP_median (ms)": round(pp_median, 2),
        "NN50 (counts)": int(nn50),
        "PNN50 (%)": round(pnn50, 2),
        "AA": round(aa, 2),
        "AA_median": round(aa_median, 2),
        "SDNN_AA": round(sdnn_aa, 2)
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
    m = 2                             # Embedding dimension
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

def extract_all_features(filtered_signal, fs=400):
    """
    Master function that strips edge artifacts, downsamples for ML compatibility,
    and extracts all parameters required for the Random Forest classifier.
    """
    # Clip the last 3 seconds to bypass IIR filter boundary transients
    stable_signal = filtered_signal[:-int(3 * fs)]
    
    # 1. Extract PRV at full 400Hz resolution
    prv_features = calculate_prv_features(stable_signal, fs)
    if not prv_features:
        return None
        
    # 2. Downsample to 100Hz for Entropy features (to match ML training dataset)
    if fs > 100:
        target_len = int(len(stable_signal) * (100 / fs))
        mse_signal = signal.resample(stable_signal, target_len)
    else:
        mse_signal = stable_signal
        
    mse_features = calculate_mse_features(mse_signal)
    
    return prv_features, mse_features