import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import os
import time
import uuid
import json
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from pydub import AudioSegment
from scipy.signal import butter, lfilter
from twilio.rest import Client as TwilioClient

st.set_page_config(page_title="Heart Sound Classifier", layout="centered")

st.title("\U0001F3A7 Heart Sound Classifier")
st.write("\nRecord, denoise and analyze heart valve sounds.")

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

CASE_HISTORY_FILE = "case_history.json"
if not os.path.exists(CASE_HISTORY_FILE):
    with open(CASE_HISTORY_FILE, "w") as f:
        json.dump([], f)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def denoise_signal(data, sr):
    b, a = butter_bandpass(20, 600, sr)
    y = lfilter(b, a, data)
    return y

def plot_waveform(y, sr, title="Waveform"):
    fig, ax = plt.subplots()
    times = np.arange(len(y)) / float(sr)
    ax.plot(times, y)
    ax.set(title=title, xlabel="Time (s)", ylabel="Amplitude")
    st.pyplot(fig)

def save_case(patient_name, height, weight, files):
    with open(CASE_HISTORY_FILE, "r") as f:
        history = json.load(f)
    case = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_name": patient_name,
        "height": height,
        "weight": weight,
        "files": files
    }
    history.append(case)
    with open(CASE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def show_case_history():
    with open(CASE_HISTORY_FILE, "r") as f:
        history = json.load(f)
    st.subheader("\U0001F4D1 Case History")
    if not history:
        st.write("No previous records found.")
    for entry in reversed(history):
        st.markdown("---")
        st.write(f"**Time:** {entry['timestamp']}")
        st.write(f"**Patient:** {entry['patient_name']}")
        st.write(f"**Height:** {entry['height']} cm")
        st.write(f"**Weight:** {entry['weight']} kg")
        for fname in entry['files']:
            st.audio(fname, format="audio/wav")
            y, sr = librosa.load(fname)
            plot_waveform(y, sr, title=f"Waveform of {os.path.basename(fname)}")

st.sidebar.header("Patient Details")
patient_name = st.sidebar.text_input("Name")
height = st.sidebar.number_input("Height (cm)", min_value=0)
weight = st.sidebar.number_input("Weight (kg)", min_value=0)

st.sidebar.header("Select Valve")
valve = st.sidebar.selectbox("Which valve sound are you recording?", ["Mitral", "Tricuspid", "Aortic", "Pulmonary"])

client_settings = ClientSettings(
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

ctx = webrtc_streamer(
    key="record", mode=WebRtcMode.SENDONLY, client_settings=client_settings,
    async_processing=True
)

if "recording" not in st.session_state:
    st.session_state.recording = False

if ctx.audio_receiver:
    if st.button("Start Recording"):
        st.session_state.recording = True
    if st.button("Stop Recording"):
        st.session_state.recording = False

    if st.session_state.recording:
        audio_frames = ctx.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            audio_frame = audio_frames[0]
            audio_data = audio_frame.to_ndarray().flatten().astype(np.float32) / 32768.0
            sr = int(audio_frame.sample_rate)

            denoised = denoise_signal(audio_data, sr)
            uid = str(uuid.uuid4())
            file_path = os.path.join(UPLOAD_DIR, f"{uid}_{valve}.wav")
            sf.write(file_path, denoised, sr)
            st.success(f"Saved {valve} valve sound to {file_path}")

            if "saved_files" not in st.session_state:
                st.session_state.saved_files = []
            st.session_state.saved_files.append(file_path)

            st.audio(file_path, format="audio/wav")
            plot_waveform(denoised, sr, title=f"{valve} Valve Waveform")

if st.button("\U0001F49B Analyse"):
    if patient_name and height and weight and "saved_files" in st.session_state:
        st.success("Analysis complete. You can now save the case.")
    else:
        st.warning("Fill all patient details and record at least one valve sound.")

if st.button("Save Case"):
    if patient_name and height and weight and "saved_files" in st.session_state:
        save_case(patient_name, height, weight, st.session_state.saved_files)
        st.success("Case saved to history.")
        st.session_state.saved_files = []
    else:
        st.warning("Fill all patient details and record at least one valve sound.")

show_case_history()

st.sidebar.header("\U0001F4E9 Send Report")
send_sms = st.sidebar.checkbox("Send report via SMS")
send_whatsapp = st.sidebar.checkbox("Send report via WhatsApp")
phone_number = st.sidebar.text_input("Phone Number (with country code)")

TWILIO_SID = st.secrets["TWILIO_SID"] if "TWILIO_SID" in st.secrets else ""
TWILIO_AUTH = st.secrets["TWILIO_AUTH"] if "TWILIO_AUTH" in st.secrets else ""
TWILIO_NUMBER = st.secrets["TWILIO_NUMBER"] if "TWILIO_NUMBER" in st.secrets else ""

if send_sms or send_whatsapp:
    if TWILIO_SID and TWILIO_AUTH and TWILIO_NUMBER and phone_number:
        client = TwilioClient(TWILIO_SID, TWILIO_AUTH)
        message_body = f"Patient: {patient_name}, Height: {height}, Weight: {weight}, Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        if send_sms:
            client.messages.create(
                to=phone_number,
                from_=TWILIO_NUMBER,
                body=message_body
            )
            st.sidebar.success("SMS sent!")
        if send_whatsapp:
            client.messages.create(
                to=f"whatsapp:{phone_number}",
                from_=f"whatsapp:{TWILIO_NUMBER}",
                body=message_body
            )
            st.sidebar.success("WhatsApp message sent!")
    else:
        st.sidebar.error("Missing Twilio credentials or phone number.")
            
