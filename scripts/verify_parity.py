"""
Generate ground-truth test vectors from MNE-Python for parity verification.
Writes results to scripts/parity_vectors/ as JSON files.
"""
import json
import os
import numpy as np
import mne
from mne.filter import create_filter, notch_filter
from scipy.signal import firwin as scipy_firwin

OUT = os.path.join(os.path.dirname(__file__), "parity_vectors")
os.makedirs(OUT, exist_ok=True)

def save_json(name, data):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(data, f)

###############################################################################
# 1. LOWPASS FILTER DESIGN
###############################################################################
print("=== LOWPASS FILTER ===")

for h_freq, sfreq in [(75.0, 256.0), (40.0, 256.0), (100.0, 512.0)]:
    h = create_filter(
        None, sfreq, l_freq=None, h_freq=h_freq,
        filter_length='auto', fir_window='hamming',
        fir_design='firwin', phase='zero', verbose=False
    )
    print(f"  LP h_freq={h_freq} sfreq={sfreq}: len={len(h)}, sum={h.sum():.10f}, "
          f"h[0]={h[0]:.10e}, h[mid]={h[len(h)//2]:.10e}")
    
    # Compute transition bandwidth manually (MNE logic)
    h_trans = min(max(0.25 * h_freq, 2.0), sfreq / 2.0 - h_freq)
    n_raw = int(np.ceil(3.3 / h_trans * sfreq))
    if n_raw % 2 == 0:
        n_raw += 1
    print(f"    h_trans={h_trans}, expected_len={n_raw}")
    
    save_json(f"lowpass_{int(h_freq)}_{int(sfreq)}.json", {
        "h_freq": h_freq, "sfreq": sfreq,
        "length": len(h),
        "sum": float(h.sum()),
        "h_first": float(h[0]),
        "h_mid": float(h[len(h)//2]),
        "h_last": float(h[-1]),
        "h_first_10": [float(x) for x in h[:10]],
        "h_trans_bandwidth": h_trans,
    })

###############################################################################
# 2. BANDPASS FILTER DESIGN
###############################################################################
print("\n=== BANDPASS FILTER ===")

for l_freq, h_freq, sfreq in [(0.1, 75.0, 256.0), (1.0, 40.0, 256.0), (0.5, 100.0, 512.0)]:
    h = create_filter(
        None, sfreq, l_freq=l_freq, h_freq=h_freq,
        filter_length='auto', fir_window='hamming',
        fir_design='firwin', phase='zero', verbose=False
    )
    print(f"  BP l={l_freq} h={h_freq} sfreq={sfreq}: len={len(h)}, sum={h.sum():.10f}, "
          f"h[0]={h[0]:.10e}, h[mid]={h[len(h)//2]:.10e}")
    
    # Compute transition bandwidths
    l_trans = min(max(0.25 * l_freq, 2.0), l_freq)
    h_trans = min(max(0.25 * h_freq, 2.0), sfreq / 2.0 - h_freq)
    min_trans = min(l_trans, h_trans)
    n_raw = int(np.ceil(3.3 / min_trans * sfreq))
    if n_raw % 2 == 0:
        n_raw += 1
    print(f"    l_trans={l_trans}, h_trans={h_trans}, min_trans={min_trans}, expected_len={n_raw}")
    
    save_json(f"bandpass_{str(l_freq).replace('.','p')}_{int(h_freq)}_{int(sfreq)}.json", {
        "l_freq": l_freq, "h_freq": h_freq, "sfreq": sfreq,
        "length": len(h),
        "sum": float(h.sum()),
        "h_first": float(h[0]),
        "h_mid": float(h[len(h)//2]),
        "h_last": float(h[-1]),
        "h_first_10": [float(x) for x in h[:10]],
        "l_trans_bandwidth": l_trans,
        "h_trans_bandwidth": h_trans,
    })

###############################################################################
# 3. NOTCH FILTER DESIGN
###############################################################################
print("\n=== NOTCH FILTER ===")

for freq, sfreq in [(60.0, 256.0), (50.0, 256.0), (60.0, 512.0)]:
    # MNE notch_filter defaults: notch_widths = freq/200, trans_bandwidth = 1.0
    nw = freq / 200.0
    tb = 1.0
    
    # MNE's notch_filter calls filter_data with:
    #   lows  = [freq - nw/2 - tb/2]
    #   highs = [freq + nw/2 + tb/2]
    # Then it does band-stop: l_freq=highs, h_freq=lows
    low_edge = freq - nw / 2.0 - tb / 2.0
    high_edge = freq + nw / 2.0 + tb / 2.0
    
    h = create_filter(
        None, sfreq, l_freq=high_edge, h_freq=low_edge,
        filter_length='auto', fir_window='hamming',
        fir_design='firwin', phase='zero',
        l_trans_bandwidth=tb/2.0, h_trans_bandwidth=tb/2.0,
        verbose=False
    )
    
    print(f"  Notch freq={freq} sfreq={sfreq}: len={len(h)}, sum={h.sum():.10f}, "
          f"h[0]={h[0]:.10e}, h[mid]={h[len(h)//2]:.10e}")
    print(f"    nw={nw}, tb={tb}, low_edge={low_edge}, high_edge={high_edge}")
    
    save_json(f"notch_{int(freq)}_{int(sfreq)}.json", {
        "freq": freq, "sfreq": sfreq,
        "notch_width": nw,
        "trans_bandwidth": tb,
        "length": len(h),
        "sum": float(h.sum()),
        "h_first": float(h[0]),
        "h_mid": float(h[len(h)//2]),
        "h_last": float(h[-1]),
        "h_first_10": [float(x) for x in h[:10]],
        "low_edge": low_edge,
        "high_edge": high_edge,
    })

###############################################################################
# 4. HIGHPASS FILTER DESIGN (existing, verify)
###############################################################################
print("\n=== HIGHPASS FILTER ===")

for l_freq, sfreq in [(0.5, 256.0), (1.0, 256.0), (0.1, 256.0)]:
    h = create_filter(
        None, sfreq, l_freq=l_freq, h_freq=None,
        filter_length='auto', fir_window='hamming',
        fir_design='firwin', phase='zero', verbose=False
    )
    print(f"  HP l_freq={l_freq} sfreq={sfreq}: len={len(h)}, sum={h.sum():.10f}, "
          f"h[0]={h[0]:.10e}, h[mid]={h[len(h)//2]:.10e}")
    
    save_json(f"highpass_{str(l_freq).replace('.','p')}_{int(sfreq)}.json", {
        "l_freq": l_freq, "sfreq": sfreq,
        "length": len(h),
        "sum": float(h.sum()),
        "h_first": float(h[0]),
        "h_mid": float(h[len(h)//2]),
        "h_last": float(h[-1]),
        "h_first_10": [float(x) for x in h[:10]],
    })

###############################################################################
# 5. FILTER APPLICATION (bandpass on synthetic data)
###############################################################################
print("\n=== FILTER APPLICATION ===")

np.random.seed(42)
sfreq = 256.0
n_ch, n_t = 4, 2560
data = np.random.randn(n_ch, n_t).astype(np.float64)

# Bandpass 0.1-75 Hz
h_bp = create_filter(
    data, sfreq, l_freq=0.1, h_freq=75.0,
    filter_length='auto', fir_window='hamming',
    fir_design='firwin', phase='zero', verbose=False
)
from mne.filter import _overlap_add_filter
filtered = _overlap_add_filter(data.copy(), h_bp, phase='zero')

print(f"  Filtered shape: {filtered.shape}")
print(f"  Ch0 first 5: {filtered[0,:5]}")
print(f"  Ch0 stats: mean={filtered[0].mean():.10f}, std={filtered[0].std():.10f}")

save_json("filter_application_bp.json", {
    "sfreq": sfreq, "n_ch": n_ch, "n_t": n_t,
    "filter_length": len(h_bp),
    "ch0_first_10": [float(x) for x in filtered[0, :10]],
    "ch0_last_10": [float(x) for x in filtered[0, -10:]],
    "ch0_mean": float(filtered[0].mean()),
    "ch0_std": float(filtered[0].std()),
    "ch1_mean": float(filtered[1].mean()),
})

###############################################################################
# 6. NOTCH FILTER APPLICATION
###############################################################################
print("\n=== NOTCH APPLICATION ===")

# Create signal with 60Hz component
t = np.arange(n_t) / sfreq
signal_60hz = np.sin(2 * np.pi * 60 * t) * 10
data_notch = np.random.randn(2, n_t).astype(np.float64)
data_notch[0] += signal_60hz
data_notch[1] += signal_60hz

h_notch = create_filter(
    data_notch, sfreq,
    l_freq=60.0 + 60.0/400.0 + 0.5,
    h_freq=60.0 - 60.0/400.0 - 0.5,
    l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
    filter_length='auto', fir_window='hamming',
    fir_design='firwin', phase='zero', verbose=False
)
filtered_notch = _overlap_add_filter(data_notch.copy(), h_notch, phase='zero')

print(f"  Notch filter len: {len(h_notch)}")
print(f"  Before 60Hz power: {np.abs(np.fft.rfft(data_notch[0]))[int(60*n_t/sfreq)]:.4f}")
print(f"  After 60Hz power: {np.abs(np.fft.rfft(filtered_notch[0]))[int(60*n_t/sfreq)]:.4f}")

save_json("notch_application.json", {
    "filter_length": len(h_notch),
    "ch0_mean": float(filtered_notch[0].mean()),
    "ch0_std": float(filtered_notch[0].std()),
})

###############################################################################
# 7. CHANNEL-WISE Z-SCORE
###############################################################################
print("\n=== CHANNEL-WISE Z-SCORE ===")

np.random.seed(123)
data_z = np.random.randn(8, 512).astype(np.float32) * 50 + np.arange(8)[:, None] * 10

# LUNA-style channel-wise z-score
mean = data_z.mean(axis=1, keepdims=True)
std = data_z.std(axis=1, keepdims=True)
eps = 1e-8
data_normed = (data_z - mean) / (std + eps)

print(f"  Ch0 mean after: {data_normed[0].mean():.10f}")
print(f"  Ch0 std after:  {data_normed[0].std():.10f}")
print(f"  Ch7 mean after: {data_normed[7].mean():.10f}")
print(f"  Ch7 std after:  {data_normed[7].std():.10f}")
# Note: numpy std uses ddof=0 by default

save_json("zscore_channelwise.json", {
    "input_seed": 123,
    "n_ch": 8, "n_t": 512,
    "ch0_first_5_normed": [float(x) for x in data_normed[0, :5]],
    "ch0_mean": float(data_normed[0].mean()),
    "ch0_std": float(data_normed[0].std()),
    "ch7_mean": float(data_normed[7].mean()),
    "ch7_std": float(data_normed[7].std()),
    "ch0_original_mean": float(mean[0, 0]),
    "ch0_original_std": float(std[0, 0]),
})

###############################################################################
# 8. EDF READING (generate a test EDF file)
###############################################################################
print("\n=== EDF FILE ===")

try:
    # Create a small test EDF
    sfreq_edf = 256.0
    n_ch_edf = 3
    n_t_edf = 2560  # 10 seconds
    ch_names_edf = ['EEG FP1-REF', 'EEG F7-REF', 'EEG T3-REF']
    ch_types_edf = ['eeg'] * n_ch_edf
    
    info = mne.create_info(ch_names_edf, sfreq_edf, ch_types_edf)
    np.random.seed(77)
    raw_data = np.random.randn(n_ch_edf, n_t_edf) * 1e-5  # ~10uV
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    
    edf_path = os.path.join(OUT, "test.edf")
    raw.export(edf_path, fmt='edf', overwrite=True, verbose=False)
    
    # Re-read it with MNE
    raw2 = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data2 = raw2.get_data()
    
    print(f"  Written EDF: {n_ch_edf} ch, {n_t_edf} samples @ {sfreq_edf} Hz")
    print(f"  Re-read shape: {data2.shape}")
    print(f"  Ch0 first 5: {data2[0,:5]}")
    print(f"  Ch0 mean: {data2[0].mean():.10e}")
    print(f"  Ch names: {raw2.ch_names}")
    
    save_json("edf_read.json", {
        "n_ch": data2.shape[0],
        "n_t": data2.shape[1],
        "sfreq": float(raw2.info['sfreq']),
        "ch_names": raw2.ch_names,
        "ch0_first_5": [float(x) for x in data2[0, :5]],
        "ch0_mean": float(data2[0].mean()),
        "ch0_std": float(data2[0].std()),
        "ch1_mean": float(data2[1].mean()),
    })
    
except Exception as e:
    print(f"  EDF test skipped: {e}")

###############################################################################
# 9. BANDPASS FILTER - DETAILED FIRWIN COMPARISON
###############################################################################
print("\n=== DETAILED FIRWIN COMPARISON ===")

# Test MNE's _firwin_design directly
from mne.filter import _firwin_design

# For bandpass 0.1-75 Hz at 256 Hz
sfreq = 256.0
l_freq, h_freq = 0.1, 75.0
l_trans = min(max(0.25 * l_freq, 2.0), l_freq)
h_trans = min(max(0.25 * h_freq, 2.0), sfreq / 2.0 - h_freq)

# The frequency/gain arrays for bandpass
l_stop = l_freq - l_trans
h_stop = h_freq + h_trans
freq = [0, l_stop, l_freq, h_freq, h_stop, sfreq/2.0]
gain = [0, 0, 1, 1, 0, 0]

# Normalize to [0,1]
freq_norm = np.array(freq) / (sfreq / 2.0)

# Filter length
min_trans = min(l_trans, h_trans)
n = int(np.ceil(3.3 / min_trans * sfreq))
if n % 2 == 0:
    n += 1

print(f"  Bandpass 0.1-75 at 256Hz:")
print(f"    l_trans={l_trans}, h_trans={h_trans}, min_trans={min_trans}")
print(f"    l_stop={l_stop}, h_stop={h_stop}")
print(f"    freq={freq}, gain={gain}")
print(f"    freq_norm={list(freq_norm)}")
print(f"    N={n}")

h = _firwin_design(n, freq_norm, gain, 'hamming', sfreq=sfreq)
print(f"    h.sum()={h.sum():.15f}")
print(f"    h[0]={h[0]:.15e}")
print(f"    h[N//2]={h[n//2]:.15e}")
print(f"    h[:5]={list(h[:5])}")

save_json("firwin_bandpass_detailed.json", {
    "N": n,
    "l_stop": l_stop,
    "h_stop": h_stop,
    "l_trans": l_trans,
    "h_trans": h_trans,
    "freq": freq,
    "gain": gain,
    "h_sum": float(h.sum()),
    "h_first_20": [float(x) for x in h[:20]],
    "h_mid_5": [float(x) for x in h[n//2-2:n//2+3]],
    "h_last_20": [float(x) for x in h[-20:]],
})

###############################################################################
# 10. NOTCH FILTER - DETAILED
###############################################################################
print("\n=== DETAILED NOTCH COMPARISON ===")

freq_notch = 60.0
sfreq = 256.0
nw = freq_notch / 200.0  # = 0.3
tb = 1.0

# MNE notch_filter:
# lows = [freq - nw/2 - tb/2]  = [60 - 0.15 - 0.5] = [59.35]
# highs = [freq + nw/2 + tb/2] = [60 + 0.15 + 0.5] = [60.65]
# Then calls filter_data(x, sfreq, highs, lows, ..., tb/2, tb/2)
# i.e. l_freq=highs=60.65, h_freq=lows=59.35  (bandstop)
# 
# In _triage_filter_params for bandstop (reverse=True):
#   f_s1 = h_freq = 59.35          (stop edge below notch)
#   f_s2 = l_freq = 60.65          (stop edge above notch)
#   f_p1 = h_freq - h_trans = 59.35 - 0.5 = 58.85  (pass edge below)
#   f_p2 = l_freq + l_trans = 60.65 + 0.5 = 61.15  (pass edge above)
#
# freq array: [0, f_p1, f_s1, f_s2, f_p2, nyq]
# gain array: [1, 1,    0,    0,    1,    1]

f_s1 = 59.35
f_s2 = 60.65
f_p1 = 58.85
f_p2 = 61.15
nyq = sfreq / 2.0

freq_arr = [0, f_p1, f_s1, f_s2, f_p2, nyq]
gain_arr = [1, 1, 0, 0, 1, 1]
freq_norm = np.array(freq_arr) / nyq

# Filter length from the transition bandwidth (0.5 Hz)
n = int(np.ceil(3.3 / 0.5 * sfreq))
if n % 2 == 0:
    n += 1
print(f"  Notch 60Hz at 256Hz:")
print(f"    nw={nw}, tb={tb}")
print(f"    f_s1={f_s1}, f_s2={f_s2}, f_p1={f_p1}, f_p2={f_p2}")
print(f"    freq={freq_arr}, gain={gain_arr}")
print(f"    N={n}")

h_notch_detail = _firwin_design(n, freq_norm, gain_arr, 'hamming', sfreq=sfreq)
print(f"    h.sum()={h_notch_detail.sum():.15f}")
print(f"    h[0]={h_notch_detail[0]:.15e}")
print(f"    h[N//2]={h_notch_detail[n//2]:.15e}")

save_json("firwin_notch_detailed.json", {
    "N": n,
    "freq": freq_arr,
    "gain": gain_arr,
    "h_sum": float(h_notch_detail.sum()),
    "h_first_20": [float(x) for x in h_notch_detail[:20]],
    "h_mid_5": [float(x) for x in h_notch_detail[n//2-2:n//2+3]],
    "h_last_20": [float(x) for x in h_notch_detail[-20:]],
})

print("\n=== ALL VECTORS SAVED ===")
