import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt
import socket

ip = "192.168.0.100"
port = 4076

def receive_ecg_data(ip, port, num_samples=4000):
    data_list = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((ip, port))
        print(f"Connected to ESP32 at {ip}:{port}")

        buffer = ""
        while len(data_list) < num_samples:
            data = s.recv(1024)  # receive bytes
            if not data:
                break

            buffer += data.decode('utf-8')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                try:
                    value = float(line.strip())
                    data_list.append(value)
                    if len(data_list) >= num_samples:
                        break
                except ValueError:
                    # Ignore lines that can't be converted
                    pass

    ecg_array = np.array(data_list)
    print(f"Received {len(ecg_array)} samples")
    return ecg_array




    
def upload_and_process_ecg(ecg_data):
    ecg_file = ecg_data
    
    if ecg_file is not None:
        #ecg_data = pd.read_csv(ecg_file).to_numpy()
        
        ecg_rate = 250
        
        time_ecg = np.linspace(0, len(ecg_data)/ecg_rate,len(ecg_data))

        # ECG Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_ecg, ecg_data, label="ECG Signal")
        ax.set_title("ECG Signal Visualization")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        ax.legend()
        plt.show()


        
        return ecg_data
    return None

def process_ecg(ecg_data,ecg_rate):
    ecg_data = ecg_data.flatten()
    signals, info = nk.ecg_process(ecg_data, sampling_rate=ecg_rate)
    r_peaks = info["ECG_R_Peaks"]

    r_sec=r_peaks/ecg_rate
    delineate_signal, delineate_info = nk.ecg_delineate(ecg_data, rpeaks=r_peaks, sampling_rate=ecg_rate, method="dwt")

    # Extract P-Peaks P-wave onset and offset
    p_onsets = delineate_info["ECG_P_Onsets"]
    p_offsets = delineate_info["ECG_P_Offsets"]
    p_peaks = delineate_info["ECG_P_Peaks"]
    
    # Extract R onset and offset
    r_onsets = delineate_info["ECG_R_Onsets"]
    r_offsets = delineate_info["ECG_R_Offsets"]
    
    #print(q_peaks)
    #print(r_onsets)
    #print(p_offsets)
    
    # Extract T-Peaks T-wave onset and offset
    t_onsets = delineate_info["ECG_T_Onsets"]
    t_offsets = delineate_info["ECG_T_Offsets"]
    t_peaks = delineate_info["ECG_T_Peaks"]
    
    # Extract Q Peaks ans peaks 
    q_peaks = delineate_info["ECG_Q_Peaks"]
    s_peaks = delineate_info["ECG_S_Peaks"]

  

    
    return r_peaks, t_offsets, t_onsets, r_onsets, r_offsets, p_onsets, p_offsets, p_peaks, t_peaks

def value_analysis(t_offsets, t_onsets, r_onsets, r_offsets, p_onsets, p_offsets, r_peaks, p_peaks, t_peaks, ecg_data ):
  qt_intervals=(np.array(t_offsets)-np.array(r_onsets))
  qt_intervals = qt_intervals[~np.isnan(qt_intervals)]
  qt_interval=np.mean(qt_intervals/ecg_rate)  if len(qt_intervals) > 0 else 0

  rr_interval=np.mean(np.diff(r_peaks) / ecg_rate)
  PP_intervals= np.diff(p_peaks)/ecg_rate
  PP_intervals= PP_intervals[~np.isnan(PP_intervals)]

  pr_intervals=(np.array(r_onsets)-np.array(p_onsets))
  pr_intervals = pr_intervals[~np.isnan(pr_intervals)]
  pr_interval=np.mean(pr_intervals/ecg_rate) if len(pr_intervals) > 0 else 0

  pr_segments=(np.array(r_onsets)-np.array(p_offsets))
  pr_segments = pr_segments[~np.isnan(pr_segments)]
  pr_segment=np.mean(pr_segments/ecg_rate) if len(pr_segments) > 0 else 0

  st_segments=(np.array(t_onsets)-np.array(r_offsets))
  st_segments = st_segments[~np.isnan(st_segments)]
  st_segment=np.mean(st_segments/ecg_rate) if len(st_segments) > 0 else 0

  tp_segments=(np.array(p_onsets[1:])-np.array(t_offsets[:-1]))
  tp_segments = tp_segments[~np.isnan(tp_segments)]
  tp_segment=np.mean(tp_segments/ecg_rate) if len(tp_segments) > 0 else 0

  p_waves=(np.array(p_offsets)-np.array(p_onsets))
  p_waves = p_waves[~np.isnan(p_waves)]
  p_wave =np.mean(p_waves/ecg_rate) if len(p_waves) > 0 else 0

  qrs_waves=(np.array(r_offsets)-np.array(r_onsets))
  qrs_waves = qrs_waves[~np.isnan(qrs_waves)]
  qrs_wave =np.mean(qrs_waves/ecg_rate) if len(qrs_waves) > 0 else 0

  t_waves=(np.array(t_offsets)-np.array(t_onsets))
  t_waves= t_waves[~np.isnan(t_waves)]
  t_wave=np.mean(t_waves/ecg_rate) if len(t_waves) > 0 else 0

  r_amplitudes = ecg_data[r_peaks]
  p_amplitudes = ecg_data[p_peaks]
  t_amplitudes = ecg_data[t_peaks]

  r_amplitudes = r_amplitudes[~np.isnan(r_amplitudes)]
  p_amplitudes = p_amplitudes[~np.isnan(p_amplitudes)]
  t_amplitudes = t_amplitudes[~np.isnan(t_amplitudes)]

  p_wave_a =np.mean(p_amplitudes) if len(p_amplitudes) > 0 else 0
  t_wave_a =np.mean(t_amplitudes) if len(t_amplitudes) > 0 else 0
  r_wave_a =np.mean(r_amplitudes) if len(r_amplitudes) > 0 else 0

  return pr_intervals, qt_interval, rr_interval, pr_interval, pr_segment, st_segment, tp_segment, p_wave_a, r_wave_a, t_wave_a, p_wave, qrs_wave, t_wave,PP_intervals

def ecg_components_typical_lead(qt_interval, rr_interval, pr_interval, pr_segment, st_segment, tp_segment, p_wave_a, r_wave_a, t_wave_a, p_wave, qrs_wave, t_wave):


    # Column 1 - Intervals
    print(f"QT Interval: {qt_interval:.2f}")
    print(f"RR Interval: {rr_interval:.2f}")
    print(f"PR Interval: {pr_interval:.2f}")
    print()

    # Column 2 - Segments
    print(f"PR Segment: {pr_segment:.2f}")
    print(f"ST Segment: {st_segment:.2f}")
    print(f"TP Segment: {tp_segment:.2f}")
    print()

    # Column 3 - Amplitudes
    print(f"P Wave Amplitude: {p_wave_a:.2f}")
    print(f"R Wave Amplitude: {r_wave_a:.2f}")
    print(f"T Wave Amplitude: {t_wave_a:.2f}")
    print()

    # Column 4 - Waves
    print(f"P Wave: {p_wave:.2f}")
    print(f"QRS Wave: {qrs_wave:.2f}")
    print(f"T Wave: {t_wave:.2f}")




    
def classify_av_block(pr_interval, r_peaks, ecg_rate,pr_intervals,p_peaks,PP_intervals):
    #st.write(ecg_rate)
    rr_intervals = np.diff(r_peaks) / ecg_rate  #smapling rate
    
    if pr_interval > 0.2 and all(rr_intervals > 0.6): 
        return "First-Degree AV Block"
   # 1. Mobitz I (Wenckebach): PR interval increases then a beat is dropped (e.g., 4 P : 3 QRS)
    pr_increasing = all(pr_intervals[i] > pr_intervals[i-1] for i in range(1, len(pr_intervals)))
    p_q_ratio = len(p_peaks) / len(r_peaks)
    
    if pr_increasing and 1.2 < np.mean(rr_intervals) and round(p_q_ratio, 1) in [1.3, 1.4]:  # ~4:3
        return "Second-Degree AV Block (Mobitz I)"

    # 2. Mobitz II: PR interval is constant, but P:QRS = 2:1 or 3:1, and beats are dropped
    pr_std = np.std(pr_intervals)
    if pr_std < 0.03 and round(p_q_ratio) in [2, 3]:
        return "Second-Degree AV Block (Mobitz II)"

    # 3. Third-Degree Block: No relation between P and QRS (very low correlation)
    
    # Step 1: Truncate to same length for correlation
    min_len = min(len(PP_intervals), len(rr_intervals))
    PP_intervals_ = PP_intervals[:min_len]
    rr_intervals_ = rr_intervals[:min_len]
    
    # Step 2: Calculate correlation coefficient
    correlation = np.corrcoef(PP_intervals_, rr_intervals_)[0, 1]
    
    #print("PP intervals:", PP_intervals_)
    #print("RR intervals:", rr_intervals_)
    #print("Correlation coefficient:", correlation)
    
    # Step 3: Check for Third-Degree AV Block
    if abs(correlation) < 0.3 and len(PP_intervals)>len(rr_intervals):
        return "Possible Third-Degree AV Block: weak/no correlation between P and QRS rhythms"
    
    return "Normal or Other Rhythm"

def classify_heart_rate(heart_rate):
    if heart_rate > 100:
        return f"Tachycardia: {heart_rate}"
    if heart_rate < 60:
        return f"Bradycardia: {heart_rate}"
    return f"Normal: {heart_rate}"


def main(ecg_data):
    ecg_data = upload_and_process_ecg(ecg_data)
    ecg_data = ecg_data.flatten()
    
    if ecg_data is not None:
        r_peaks, t_offsets, t_onsets, r_onsets, r_offsets, p_onsets, p_offsets, p_peaks, t_peaks = process_ecg(ecg_data, ecg_rate)
        heart_rate = 60 / np.mean(np.diff(r_peaks) / ecg_rate)
        heart_rate_classification = classify_heart_rate(heart_rate)
        pr_intervals, qt_interval, rr_interval, pr_interval, pr_segment, st_segment, tp_segment, p_wave_a, r_wave_a, t_wave_a, p_wave, qrs_wave, t_wave,PP_intervals = value_analysis(t_offsets, t_onsets, r_onsets, r_offsets, p_onsets, p_offsets, r_peaks, p_peaks, t_peaks, ecg_data)
        classification = classify_av_block(pr_interval, r_peaks, ecg_rate,pr_intervals,p_peaks,PP_intervals)
        print("ECG Analysis Parameters")
        print()
        print(f"🩺 PR Interval: {pr_interval}")
        print()
        print()
        print("ECG Analysis Results")
        print()
        print(f"🩺 AV Block Classification: {classification}")
        print()
        print(f"🩺 Heart Rate: {heart_rate:.2f} BPM ({heart_rate_classification})")
        print()
        print()

        ecg_components_typical_lead(qt_interval, rr_interval, pr_interval, pr_segment, st_segment, tp_segment, p_wave_a, r_wave_a, t_wave_a, p_wave, qrs_wave, t_wave)
        print()



ecg_rate = 250
print('ok')
ecg_data = receive_ecg_data(ip, port)
main(ecg_data) 
