import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io.wavfile as wav
import soundfile as sf
from scipy.signal import butter, lfilter
from datetime import datetime
import json
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import io
from twilio.rest import Client

st.set_page_config(layout="wide")
st.title("💓 HEARTEST : GIRI's PCG analyzer")

UPLOAD_FOLDER = "uploaded_audios"
EDITED_FOLDER = "edited_audios"
PATIENT_DATA = "patient_data.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EDITED_FOLDER, exist_ok=True)


def save_patient_data(data):
    if os.path.exists(PATIENT_DATA):
        with open(PATIENT_DATA, "r") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.append(data)
    with open(PATIENT_DATA, "w") as f:
        json.dump(existing, f)


def load_patient_data():
    if os.path.exists(PATIENT_DATA):
        with open(PATIENT_DATA, "r") as f:
            return json.load(f)
    return []


def send_sms(phone_number, message):
    TWILIO_ACCOUNT_SID = "AC15ee7441c990e6e8a5afc996ed4a55a1"
    TWILIO_AUTH_TOKEN = "6bc0831dae8edb1753ace573a92b6337"
    TWILIO_PHONE_NUMBER = "+19096391894"
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=phone_number
    )


def reduce_noise(audio, sr, cutoff=0.05):
    b, a = butter(6, cutoff)
    return lfilter(b, a, audio)


def wav_to_bytes(audio_data, sample_rate):
    output = io.BytesIO()
    wav.write(output, sample_rate, audio_data.astype(np.int16))
    return output.getvalue()


def save_wav(path, audio_data, sample_rate):
    wav.write(path, sample_rate, audio_data.astype(np.int16))


def show_waveform(audio, sr, label, color='blue'):
    times = np.linspace(0, len(audio)/sr, num=len(audio))
    fig, ax = plt.subplots()
    ax.plot(times, audio, color=color)
    ax.set_title(f"Waveform - {label}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)


def edit_and_show_waveform(path, label, save_edit=False):
    sr, audio = wav.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    st.markdown(f"#### {label} Valve")

    col1, col2, col3 = st.columns(3)
    with col1:
        amplitude_factor = st.slider(f"{label} Amplitude", 0.1, 5.0, 1.0, key=f"amp_slider_{label}")
    with col2:
        duration_slider = st.slider(f"{label} Duration (s)", 1, int(len(audio) / sr), 5, key=f"dur_slider_{label}")
    with col3:
        noise_cutoff = st.slider(f"{label} Noise Cutoff", 0.01, 0.5, 0.05, step=0.01, key=f"noise_slider_{label}")

    adjusted_audio = audio[:duration_slider * sr] * amplitude_factor
    filtered_audio = reduce_noise(adjusted_audio, sr, cutoff=noise_cutoff)

    edited_path = os.path.join(EDITED_FOLDER, os.path.basename(path))
    if save_edit:
        save_wav(edited_path, filtered_audio, sr)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{label} Original**")
        st.audio(path, format="audio/wav")
        show_waveform(audio, sr, f"{label} (Original)", color='blue')
    with col2:
        st.write(f"**{label} Edited**")
        st.audio(io.BytesIO(wav_to_bytes(filtered_audio, sr)), format='audio/wav')
        show_waveform(filtered_audio, sr, f"{label} (Edited)", color='red')

    return edited_path


st.subheader("🎧 Upload Heart Valve Sounds")
valve_labels = ["Aortic", "Pulmonary", "Tricuspid", "Mitral"]
valve_paths = {}
cols = st.columns(4)

for i, label in enumerate(valve_labels):
    with cols[i]:
        file = st.file_uploader(f"Upload {label} Valve", type=["wav"], key=f"upload_{label}")
        if file:
            path = os.path.join(UPLOAD_FOLDER, f"{label}_{file.name}")
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            valve_paths[label] = path

if "patient_saved" not in st.session_state:
    st.session_state["patient_saved"] = False

with st.sidebar.expander("🖞️ Add Patient Info"):
    name = st.text_input("Name")
    age = st.number_input("Age", 1, 120)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0)
    weight = st.number_input("Weight (kg)", min_value=2.0, max_value=300.0)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    notes = st.text_area("Clinical Notes")
    phone = st.text_input("📲 Patient Phone (E.g. +15558675309)")

if height and weight:
    bmi = round(weight / ((height / 100) ** 2), 2)
    st.markdown(f"**BMI:** {bmi}")

if st.button("📂 Save Patient Case", type="primary"):
    if len(valve_paths) == 4:
        edited_files = []
        for label in valve_labels:
            edited_path = edit_and_show_waveform(valve_paths[label], label, save_edit=True)
            edited_files.append(os.path.basename(edited_path))

        data = {
            "name": name,
            "age": age,
            "gender": gender,
            "notes": notes,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "file": ", ".join(edited_files),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_patient_data(data)
        st.session_state["patient_saved"] = True
        st.success("Patient data saved.")

if st.session_state["patient_saved"]:
    st.subheader("🔹 Original & Edited Waveforms")
    for label in valve_labels:
        if label in valve_paths:
            edit_and_show_waveform(valve_paths[label], label)

if st.button("📤 Send Case via SMS"):
    if len(valve_paths) == 4 and phone:
        try:
            message = (
                f"🌹 PCG Case Summary\n"
                f"Name: {name}\nAge: {age}\nGender: {gender}\nBMI: {bmi}\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nNotes: {notes}"
            )
            send_sms(phone, message)
            st.success("📨 Case sent via SMS.")
        except Exception as e:
            st.error(f"❌ Failed to send SMS: {e}")
    else:
        st.warning("Please complete all uploads and phone number.")

st.subheader("📚 Case History")
patient_data = load_patient_data()
if patient_data:
    for i, entry in enumerate(patient_data[::-1]):
        with st.expander(f"📌 {entry['name']} ({entry['age']} y/o) - {entry['date']}"):
            st.write(f"Gender: {entry['gender']}")
            st.write(f"Height: {entry.get('height', 'N/A')} cm")
            st.write(f"Weight: {entry.get('weight', 'N/A')} kg")
            st.write(f"BMI: {entry.get('bmi', 'N/A')}")
            st.write(f"Notes: {entry['notes']}")
            for label in valve_labels:
                filename = f"{label}_{entry['file'].split(', ')[0].split('_', 1)[1]}"
                edited_file = os.path.join(EDITED_FOLDER, filename)
                if os.path.exists(edited_file):
                    st.audio(edited_file, format="audio/wav")
                    sr, audio = wav.read(edited_file)
                    if audio.ndim > 1:
                        audio = audio[:, 0]
                    show_waveform(audio, sr, f"{label} History")
else:
    st.info("No history records found.")

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #006400;
    color: white;
}
</style>""", unsafe_allow_html=True)
    
