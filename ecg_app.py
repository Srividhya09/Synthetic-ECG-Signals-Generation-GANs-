import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO  # For in-memory file handling

# Load model
@st.cache_resource
def load_generator_model():
    return load_model("save_model/final_generator.h5")

# Generate signals
def generate_ecg_signals(generator_model, num_signals):
    noise = np.random.normal(0, 1, (num_signals, 100))
    generated_signals = generator_model.predict(noise)
    return generated_signals

# App layout and interface
st.set_page_config(
    page_title="Synthetic ECG Signal Generator",
    page_icon='📈',
    layout="wide"
)

# Apply custom styling
st.markdown(
    """
    <style>
        /* Background for the entire app */
        .stApp {
            background-color: #6A1B9A !important; /* Deep purple */
            color: white !important; /* Text color */
        }
        /* Title and subtitles styling */
        .title-style {
            font-size: 3rem;
            font-weight: bold;
            color: #FFFFFF !important; /* White color */
            text-align: center;
            margin-top: -20px;
        }
        .subtitle-style {
            font-size: 1.5rem;
            color: #FFFFFF !important; /* White color */
            text-align: center;
        }
        .quotation {
            text-align: center;
            font-size: 1rem;
            font-style: italic;
            color: #FFFFFF !important; /* White color */
            margin-top: 10px;
        }
        /* Section header styling */
        .header-style {
            font-size: 1.5rem;
            font-weight: bold;
            color: #FFFFFF !important; /* White color */
            margin-bottom: 10px;
        }
        /* Button styling */
        .stButton > button {
            background-color: #4CAF50 !important; /* Green */
            color: #FFFFFF !important; /* White text */
            font-size: 1rem;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #45A049 !important; /* Darker green on hover */
        }
        /* Download button styling */
        .stDownloadButton > button {
            background-color: #4CAF50 !important; /* Green */
            color: #FFFFFF !important; /* White text */
            font-size: 1rem;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
        }
        .stDownloadButton > button:hover {
            background-color: #45A049 !important; /* Darker green on hover */
        }
        /* Horizontal line styling */
        hr {
            border: 1px solid #FFFFFF !important;
        }
        /* Signal plot labels */
        .signal-label {
            color: #FFFFFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with icon 📈
st.markdown('<div class="title-style">📈 Synthetic ECG Signal Generation Using GANs</div>', unsafe_allow_html=True)

# Subtitle with proper spacing
st.markdown('<div class="subtitle-style">Generating high-quality synthetic ECG signals using advanced GAN techniques</div>', unsafe_allow_html=True)

# Add a quotation
st.markdown('<div class="quotation">"Advancing cardiac research through realistic synthetic ECG signal generation."</div>', unsafe_allow_html=True)

st.markdown("---")  # Horizontal line for separation

# Store signals in session state to persist across re-renders
if "signals" not in st.session_state:
    st.session_state.signals = None

# User input section for number of signals
st.markdown('<div class="header-style">ECG Signal Generation</div>', unsafe_allow_html=True)
st.markdown('<p style="color:white;">Enter the number of signals to generate:</p>', unsafe_allow_html=True)
num_signals = st.number_input("", min_value=1, max_value=100, value=10)

# Generate Button
if st.button("Generate Signals"):
    model = load_generator_model()
    signals = generate_ecg_signals(model, num_signals)
    st.session_state.signals = signals  # Store generated signals in session state

# If signals exist, display and provide download options
if st.session_state.signals is not None:
    signals = st.session_state.signals

    st.markdown('<div class="header-style">Generated ECG Signals</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(num_signals, 1, figsize=(12, num_signals * 2))

    # Check if only one signal is generated
    if num_signals == 1:
        axes = [axes]  # Make it iterable by wrapping the single axis in a list

    for i, ax in enumerate(axes):
        ax.plot(signals[i].squeeze(), color='blue')
        ax.set_title(f"Signal {i+1}", fontsize=12, color='white')
        ax.set_ylabel("Amplitude", fontsize=10, color='white')
        ax.set_xlabel("Time", fontsize=10, color='white')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(pad=3.0)  # Adjust padding for clarity
    st.pyplot(fig)

    # Option to download signals as image file (PNG or JPG)
    st.markdown("---")  # Horizontal line for separation
    st.write("Download the generated signals as an image file (e.g., PNG, JPG):")

    # Save the figure to a BytesIO buffer as a PNG image
    buffer = BytesIO()
    fig.savefig(buffer, format="png")  # You can change 'png' to 'jpg' or 'jpeg' if needed
    buffer.seek(0)  # Move to the start of the buffer for downloading

    st.download_button(
        label="Download Signals as PNG",
        data=buffer,
        file_name="synthetic_ecg_signals.png",  # You can change the file extension as needed
        mime="image/png",
        use_container_width=True,
    )

    # Option to download signals as .npy file
    st.markdown("---")  # Horizontal line for separation
    st.write("Download the generated signals as a .npy file:")

    # Save signals to a file (e.g., NumPy binary)
    npy_buffer = BytesIO()
    np.save(npy_buffer, signals)
    npy_buffer.seek(0)

    st.download_button(
        label="Download Synthetic Signals as .npy",
        data=npy_buffer,
        file_name="synthetic_ecg_signals.npy",
        mime="application/octet-stream",
        use_container_width=True,
    )

# Add a section to upload and visualize downloaded signals
st.markdown('<div class="header-style">Upload and View Signals</div>', unsafe_allow_html=True)
st.markdown('<p style="color:white;">Upload a .npy file to view signals</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["npy"])

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    # Load the uploaded file
    signals = np.load(uploaded_file)
    
    # Display the signals
    st.markdown('<div class="header-style">Uploaded ECG Signals</div>', unsafe_allow_html=True)
    num_signals = signals.shape[0]
    fig, axes = plt.subplots(num_signals, 1, figsize=(12, num_signals * 2))

    # Check if only one signal is uploaded
    if num_signals == 1:
        axes = [axes]  # Make it iterable by wrapping the single axis in a list

    for i, ax in enumerate(axes):
        ax.plot(signals[i].squeeze(), color='blue')
        ax.set_title(f"Signal {i + 1}", fontsize=12, color='white')
        ax.set_ylabel("Amplitude", fontsize=10, color='white')
        ax.set_xlabel("Time", fontsize=10, color='white')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(pad=3.0)
    st.pyplot(fig)
