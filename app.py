import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import scipy.optimize as opt
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Cardiac Health Monitoring System",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
    <style>
        body {
            background-image: url('https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.pinterest.com%2Fpin%2Ffree-stock-photo-of-human-heart-anatomical-rendering-on-dark-background--699535754615618586%2F&psig=AOvVaw1DAq5FVQDWOcCUPnFRa00a&ust=1742838942829000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCJjU-5PjoIwDFQAAAAAdAAAAABAE'); /* Add your own background image */
            background-size: cover;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #1f1f1f;
            color: white;
        }
        h1 {
            font-family: 'Roboto', sans-serif;
            color: #ff6f61;
        }
        h2, h3 {
            color: #ff6f61;
        }
        .stButton>button {
            background-color: #ff6f61;
            color: white;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #ff3b30;
        }
        .stTextInput>div>input {
            background-color: #333;
            color: white;
        }
        .stFileUploader>label {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)



with st.sidebar:
    st.image("ecg_2.jpg", width=200)
    selected = option_menu(
        "Menu", ["Home", "ECG & Typical Values", "About", "Contact"],
        icons=["house", "info-circle", "info-circle", "envelope"],
        menu_icon="cast", default_index=0
    )
#st.title("📊 Cardiac Health Monitoring System")
#st.image("heart.jpg", use_container_width=True)
#ecg_rate = st.number_input("Enter ECG Sampling Rate", min_value=1, value=250)
#ppg_rate = st.number_input("Enter PPG Sampling Rate", min_value=1, value=250)
        
def upload_and_process_ecg():
    #st.image("heart.jpg", use_container_width=True)
    ecg_file = st.file_uploader("Upload ECG CSV File", type="csv")
    ppg_file = st.file_uploader("Upload PPG CSV File", type="csv")
    
    if ecg_file is not None and ppg_file is not None:
        ecg_data = pd.read_csv(ecg_file).to_numpy()
        ppg_data = pd.read_csv(ppg_file).to_numpy()
        
        #ecg_rate = st.number_input("Enter ECG Sampling Rate", min_value=1, value=250)
        #ppg_rate = st.number_input("Enter PPG Sampling Rate", min_value=1, value=250)
        #maximum=min(len(ecg_data),len(ppg_data))
        
        time_ecg = np.linspace(0, len(ecg_data)/ecg_rate,len(ecg_data))
        time_ppg = np.linspace(0, len(ecg_data)/ecg_rate,len(ppg_data))
        
        
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_ecg, ecg_data, label="ECG Signal")
        ax.set_title("ECG Signal Visualization")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        st.pyplot(fig)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_ppg, ppg_data, label="PPG Signal", color='red')
        ax.set_title("PPG Signal Visualization")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid()
        st.pyplot(fig)
        
        return ecg_data, ppg_data  #, ecg_rate, ppg_rate
    return None, None #, None, None

def process_ecg_ppg(ecg_data, ppg_data, ecg_rate, ppg_rate):
    #st.write(ecg_rate)
    ecg_data = ecg_data.flatten()
    #if ecg_data.ndim > 1:
        #ecg_data = ecg_data[:, 0]
    signals, info = nk.ecg_process(ecg_data, sampling_rate=ecg_rate)
    r_peaks = info["ECG_R_Peaks"]

    r_sec=r_peaks/ecg_rate
    #st.write(r_peaks)
    # Delineate ECG waves (find onsets and offsets)
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

    ppg_data=ppg_data.flatten()
    _, ppg_info = nk.ppg_process(ppg_data, sampling_rate=ppg_rate)

    # Get PPG peaks (pulse waves)
    ppg_peaks = ppg_info["PPG_Peaks"]

    ppg_sec=ppg_peaks/ppg_rate

    # Compute Pulse Transit Time (PTT) - Time difference between R-peaks and PPG peaks
    ptt_values = np.array([ppg_sec[i] - r_sec[i] for i in range(min(len(r_sec), len(ppg_sec)))])    # Convert to seconds
    ptt_value=np.mean(ptt_values)

    pr_intervals = np.array(r_onsets) - np.array(p_onsets)
    pr_intervals = pr_intervals[~np.isnan(pr_intervals)]
    pr_interval = np.mean(pr_intervals / ecg_rate) if len(pr_intervals) > 0 else 0
    
    return pr_interval, r_peaks, ptt_value, t_offsets, t_onsets, r_onsets, r_offsets, p_onsets, p_offsets, p_peaks, t_peaks

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

  p_wave_a =np.mean(r_amplitudes) if len(r_amplitudes) > 0 else 0
  t_wave_a =np.mean(p_amplitudes) if len(p_amplitudes) > 0 else 0
  r_wave_a =np.mean(t_amplitudes) if len(t_amplitudes) > 0 else 0

  return qt_interval, rr_interval, pr_interval, pr_segment, st_segment, tp_segment, p_wave_a, r_wave_a, t_wave_a, p_wave, qrs_wave, t_wave,PP_intervals

def ecg_components_typical_lead(qt_interval, rr_interval, pr_interval, pr_segment, st_segment, tp_segment, p_wave_a, r_wave_a, t_wave_a, p_wave, qrs_wave, t_wave):

    st.write("")

    st.subheader("📊 ECG Components Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"<div style='background-color:#1709db; padding:10px; border-radius:10px;'>"
                    f"<strong>QT Interval:</strong> {qt_interval:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#1709db; padding:10px; border-radius:10px;'>"
                    f"<strong>RR Interval:</strong> {rr_interval:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#1709db; padding:10px; border-radius:10px;'>"
                    f"<strong>PR Interval:</strong> {pr_interval:.2f}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div style='background-color:#db0909; padding:10px; border-radius:10px;'>"
                    f"<strong>PR Segment:</strong> {pr_segment:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#db0909; padding:10px; border-radius:10px;'>"
                    f"<strong>ST Segment:</strong> {st_segment:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#db0909; padding:10px; border-radius:10px;'>"
                    f"<strong>TP Segment:</strong> {tp_segment:.2f}</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div style='background-color:#10db09; padding:10px; border-radius:10px;'>"
                    f"<strong>P Wave Amplitude:</strong> {p_wave_a:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#10db09; padding:10px; border-radius:10px;'>"
                    f"<strong>R Wave Amplitude:</strong> {r_wave_a:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#10db09; padding:10px; border-radius:10px;'>"
                    f"<strong>T Wave Amplitude:</strong> {t_wave_a:.2f}</div>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"<div style='background-color:#dba709; padding:10px; border-radius:10px;'>"
                    f"<strong>P Wave:</strong> {p_wave:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#dba709; padding:10px; border-radius:10px;'>"
                    f"<strong>QRS Wave:</strong> {qrs_wave:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color:#dba709; padding:10px; border-radius:10px;'>"
                    f"<strong>T Wave:</strong> {t_wave:.2f}</div>", unsafe_allow_html=True)

    st.markdown("---")

    
def classify_av_block(pr_interval, r_peaks, ecg_rate,pr_intervals,r_peaks,p_peaks,PP_intervals):
    #st.write(ecg_rate)
    rr_intervals = np.diff(r_peaks) / ecg_rate  #smapling rate
    
    if pr_interval > 0.2 and all(rr_intervals > 0.6): 
        return "First-Degree AV Block"
   # 1. Mobitz I (Wenckebach): PR interval increases then a beat is dropped (e.g., 4 P : 3 QRS)
    pr_increasing = all(pr_intervals[i] > pr_intervals[i-1] for i in range(1, len(pr_intervals)))
    p_q_ratio = len(r_peaks) / len(r_peaks)
    
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
    if abs(correlation) < 0.3 and len((PP_intervals)>len(rr_intervals):
        return "Possible Third-Degree AV Block: weak/no correlation between P and QRS rhythms"
    
    return "Normal or Other Rhythm"

def classify_heart_rate(heart_rate):
    if heart_rate > 100:
        return f"Tachycardia: {heart_rate}"
    if heart_rate < 60:
        return f"Bradycardia: {heart_rate}"
    return f"Normal: {heart_rate}"

def calibrate():
    # Calibration for PTT, SBP, and DBP using uploaded files
    ptt_file_c = st.file_uploader("Upload PTT Calibration CSV File", type="csv")
    sbp_file_c = st.file_uploader("Upload SBP Calibration CSV File", type="csv")
    dbp_file_c = st.file_uploader("Upload DBP Calibration CSV File", type="csv")
    
    if ptt_file_c is not None and sbp_file_c is not None and dbp_file_c is not None:
        ptt_values_c = pd.read_csv(ptt_file_c).to_numpy()
        sbp_values_c = pd.read_csv(sbp_file_c).to_numpy()
        dbp_values_c = pd.read_csv(dbp_file_c).to_numpy()
        
        ptt_values_c = ptt_values_c.flatten()
        sbp_values_c = sbp_values_c.flatten()
        dbp_values_c = dbp_values_c.flatten()
        #st.write(f"Is array_1d 1D? {dbp_values_c.ndim == 1}")
        #st.write(ptt_values_c)
        #st.write(f"Is array_1d 1D? {ptt_values_c.ndim == 1}")
        #st.write(f"Is array_1d 1D? {sbp_values_c.ndim == 1}")

        #if np.any(np.isnan(ptt_values_c)) or np.any(np.isinf(ptt_values_c)):
            #st.write("ptt_values_c contains NaN or inf values")
        #else:
            #st.write("ok")
        #st.write(type(ptt_values_c)) 

        #if np.any(np.vectorize(lambda x: not isinstance(x, (int, float)))(ptt_values_c)):
            #st.write("ptt_values_c contains non-numeric values")
        #else:
            #st.write("ptt_values_c contains only numeric values")
        #st.write(ptt_values_c.shape)  # Should print (n,) where n > 1'''
    

        
        def sbp_model(ptt, a, b, c):
            return a * (ptt ** -b) + c

        def dbp_model(ptt, d, e, f):
            return d * (ptt ** -e) + f
            
        # Fit the models to find a, b, c (for SBP) and d, e, f (for DBP)
        try:
            params_sbp, _ = opt.curve_fit(sbp_model, ptt_values_c, sbp_values_c, p0=[1, 1, 100], maxfev=10000, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            params_dbp, _ = opt.curve_fit(dbp_model, ptt_values_c, dbp_values_c, p0=[1, 1, 60], maxfev=10000, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            a, b, c = params_sbp
            d, e, f = params_dbp

            #st.write(f"Calibration successful! SBP model: SBP = {a:.2f} * PTT^(-{b:.2f}) + {c:.2f}")
            #st.write(f"Calibration successful! DBP model: DBP = {d:.2f} * PTT^(-{e:.2f}) + {f:.2f}")

            st.write(f"✅ **:green[Calibration successful!]**")
            st.write(f"**SBP model:** SBP = :violet[{a:.2f}] * PTT^(-:red[{b:.2f}]) + :blue[{c:.2f}]")

            # Add vertical space
            st.write("")
            st.write("")  # More spaces if needed
            
            st.write(f"✅ **:orange[Calibration successful!]**")
            st.write(f"**DBP model:** DBP = :violet[{d:.2f}] * PTT^(-:red[{e:.2f}]) + :blue[{f:.2f}]")
        except Exception as e:
            st.write("Error during curve fitting:", e)


       

        #st.write(f"Calibration successful! SBP model: SBP = {a:.2f} * PTT^(-{b:.2f}) + {c:.2f}")
        #st.write(f"Calibration successful! DBP model: DBP = {d:.2f} * PTT^(-{e:.2f}) + {f:.2f}")
        
        return a, b, c, d, e, f
    else:
        st.warning("Please upload the calibration files to proceed.")
        return None, None, None, None, None, None

def main():
    ecg_data, ppg_data  = upload_and_process_ecg() #, ecg_rate, ppg_rate
    
    if ecg_data is not None:
        pr_interval, r_peaks, ptt_value, t_offsets, t_onsets, r_onsets, r_offsets, p_onsets, p_offsets, p_peaks, t_peaks, = process_ecg_ppg(ecg_data, ppg_data, ecg_rate, ppg_rate )
        heart_rate = 60 / np.mean(np.diff(r_peaks) / ecg_rate)
        heart_rate_classification = classify_heart_rate(heart_rate)
        qt_interval, rr_interval, pr_interval, pr_segment, st_segment, tp_segment, p_wave_a, r_wave_a, t_wave_a, p_wave, qrs_wave, t_wave,PP_intervals = value_analysis(t_offsets, t_onsets, r_onsets, r_offsets, p_onsets, p_offsets, r_peaks, p_peaks, t_peaks, ecg_data)
        classification = classify_av_block(pr_interval, r_peaks, ecg_rate,p_peaks,PP_intervals)
        st.subheader("ECG Analysis Parameters")
        st.write("")
        st.write(f"  🩺PR Interval : :blue[{pr_interval}]")
        st.write("")
        st.write("")
        st.subheader("**ECG Analysis Results**")
        st.write("")
        st.write(f"  🩺AV Block Classification: :green[{classification}]")
        st.write("")
        st.write(f"   🩺Heart Rate: :red[{heart_rate:.2f}] BPM 🩺 (:blue[{heart_rate_classification}])")
        st.write("")
        st.write("")
        ecg_components_typical_lead(qt_interval, rr_interval, pr_interval, pr_segment, st_segment, tp_segment, p_wave_a, r_wave_a, t_wave_a, p_wave, qrs_wave, t_wave)
        st.write("")

        if pr_interval == 0:
            st.warning("Calibration Required: Please check ECG signal quality.")
        
        a, b, c, d, e, f = calibrate()  # Calibrate for PTT, SBP, DBP
        
        if a is not None and b is not None and c is not None:
            #ptt_value = np.mean(np.diff(ppg_data))  # Example for PTT calculation (replace with actual calculation)
            predicted_sbp = a * (ptt_value ** -b) + c
            predicted_dbp = d * (ptt_value ** -e) + f
            st.write(f"Predicted Blood Pressure: :red[{predicted_sbp:.2f}/{predicted_dbp:.2f} mmHg]")
        
        #st.sidebar.header("Contact")
        #st.sidebar.write("For support, contact: Tamil | Email: tamil@example.com")


if selected == "Home":
    #st.title("📊 Cardiac Health Monitoring System")
    st.title("📊 Cardiac Health Monitoring System")
    st.image("heart.jpg", use_container_width=True)
    ecg_rate = st.number_input("Enter ECG Sampling Rate", min_value=1, value=250)
    ppg_rate = st.number_input("Enter PPG Sampling Rate", min_value=1, value=250)
        
    if __name__ == "__main__":
        main() 
elif selected == "ECG & Typical Values":
    st.title("ECG & Typical Values")
    st.subheader("🩺 **Annotation of ECG Componets and its Typical Values**")
    st.image("Anotation.png", width=500)
    st.image("values.jpg", width=500)
# About Section
elif selected == "About":
    st.title("📌 About")

    st.subheader("🩺 **Advanced ECG & PPG Signal Analysis for Cardiovascular Health**")
    
    st.write(
        "This project focuses on the **real-time analysis** of :blue[ECG (Electrocardiogram)] and "
        ":green[PPG (Photoplethysmogram)] signals, enabling early detection of **cardiac abnormalities** "
        "and **personalized blood pressure estimation**. It is designed for **healthcare professionals** "
        "and **researchers** to enhance cardiovascular assessments using **digital signal processing (DSP)** "
        "and **biomedical engineering advancements**."
    )
    
    st.subheader("🔍 **Key Features & Capabilities:**")
    st.write("- ✅ **Heart Rate Analysis:** Detects **Tachycardia** (⚠️ High Heart Rate) & **Bradycardia** (⚠️ Low Heart Rate)")
    st.write("- ✅ **ECG Interval Analysis:** Examines **PR Interval, QRS Complex, and QT Interval** for cardiac health")
    st.write("- ✅ **AV Nodal Blockage Detection:** Identifies **Atrioventricular (AV) block conditions** for early diagnosis")
    st.write("- ✅ **ECG-PPG Calibration:** Utilizes **individualized calibration** to estimate **Blood Pressure (BP)**")
    st.write("- ✅ **Digital Signal Processing (DSP):** Leverages **advanced filtering & feature extraction techniques**")
    
    st.subheader("🔬 **Future Advancements:**")
    st.write("- 🚀 **Large-scale ECG & PPG Analysis** for population-based cardiovascular research")
    st.write("- 🚀 **Large-scale ECG & PPG Analysis** for population-based cardiovascular research")
    st.write("- 🧠 **AI-Driven Predictions** using **Deep Learning & Machine Learning models**")
    st.write("- 📡 **Cloud & IoT Integration** for remote health monitoring and real-time analysis")
    
    st.write(
        "**This system aims to revolutionize biosignal processing by enhancing accuracy, efficiency, "
        "and real-time diagnostics for cardiovascular health monitoring.**"
    )


# Contact Section
elif selected == "Contact":
    st.title("📞 Contact")
    st.write(
        "For inquiries, collaborations, or feedback, please reach out to us through the following channels:" 
    )
    st.write("- **Email:** thamilezaiananthakumar@gmail.com")
    st.write("- **Phone:** +940762934089")
    st.write("- **GitHub:** [Thamilezai Ananthakumar](https://github.com/ThamilezaiAnanthakumar)")
    st.write("- **LinkedIn:** [Thamilezai Ananthakumar](https://www.linkedin.com/in/thamilezai-ananthakumar-387a922a4)")

# Footer
st.markdown("""
---
Developed with ❤️ by Thamilezai Ananthakumar
""")
