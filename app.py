import streamlit as st
import joblib
import pandas as pd
import re
from PIL import Image
import pytesseract
from datetime import datetime
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="TruthLens AI", layout="wide")

# --------------------------------------------------
# NEWSPAPER STYLE CSS
# --------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Libre+Baskerville&display=swap');

.stApp {
    background-color: #fdf6e3;
}

.block-container {
    background-color: rgba(255,255,255,0.96);
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.15);
}

h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 52px !important;
    text-align: center;
    border-bottom: 4px double black;
    padding-bottom: 10px;
}

h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

html, body, [class*="css"] {
    font-family: 'Libre Baskerville', serif !important;
    font-size: 17px;
}

.result-card {
    padding: 30px;
    border-radius: 12px;
    text-align: center;
}

.stButton>button {
    border-radius: 6px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1>TruthLens AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-Powered News Credibility Analyzer</p>", unsafe_allow_html=True)

st.markdown("""
<div style="background:#111;color:white;padding:8px;text-align:center;font-size:14px;">
WORLD | POLITICS | BUSINESS | TECHNOLOGY | HEALTH | SCIENCE
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#b30000;color:white;padding:8px;font-weight:bold;text-align:center;">
🚨 BREAKING: AI-driven misinformation detection is transforming digital journalism.
</div>
""", unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# LAYOUT
# --------------------------------------------------
left, right = st.columns([2,1])

with left:
    st.subheader("📝 News Article Input")
    text = st.text_area(
        "Paste News Article Here",
        height=350
    )

with right:
    st.subheader("📊 Model Info")
    st.markdown("""
    - Model: TF-IDF + Logistic Regression  
    - Calibration: Platt Scaling  
    - Conservative Threshold: 0.75  
    - OCR Enabled  
    """)

# --------------------------------------------------
# OCR
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload News Screenshot (OCR)", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    extracted_text = pytesseract.image_to_string(image)
    text = extracted_text
    st.success("Text Extracted Successfully!")

# --------------------------------------------------
# ANALYZE
# --------------------------------------------------
if st.button("Analyze News"):

    if text.strip() == "":
        st.warning("Please enter news content.")
    elif len(text.split()) < 10:
        st.warning("Please enter a longer news article for better accuracy.")
    else:

        cleaned = clean_text(text)

        with st.spinner("Analyzing with AI Model..."):

            prob_fake = model.predict_proba([cleaned])[0][1]
            prob_real = 1 - prob_fake

            if prob_fake > 0.75:
                label = "FAKE NEWS"
                color = "red"
            elif prob_fake < 0.25:
                label = "REAL NEWS"
                color = "green"
            else:
                label = "UNCERTAIN"
                color = "orange"

            confidence = (max(prob_fake, prob_real) - 0.5) * 2 * 100
            confidence = max(0, min(confidence, 100))
            confidence = round(confidence, 2)

        # RESULT
        st.markdown(f"""
        <div style="border:3px solid {color};padding:30px;text-align:center;margin-top:20px;">
            <h1 style="color:{color};">{label}</h1>
            <h3>Confidence Level: {confidence}%</h3>
        </div>
        """, unsafe_allow_html=True)

        # Confidence explanation
        if confidence < 40:
            st.info("Low confidence prediction. Please verify manually.")
        elif confidence < 70:
            st.info("Moderate confidence prediction.")
        else:
            st.success("High confidence prediction.")

        st.divider()

        # DASHBOARD
        st.subheader("AI Risk Analysis Dashboard")
        col1, col2 = st.columns(2)

        with col1:
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_real * 100,
                title={'text': "Credibility Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"}
                }
            ))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_fake * 100,
                title={'text': "Fake Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"}
                }
            ))
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # SAVE HISTORY
        history_entry = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Prediction": label,
            "Confidence (%)": confidence,
            "Fake Probability (%)": round(prob_fake*100, 2),
            "Real Probability (%)": round(prob_real*100, 2)
        }

        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append(history_entry)

# --------------------------------------------------
# HISTORY
# --------------------------------------------------
if "history" in st.session_state and st.session_state.history:
    st.subheader("📜 Analysis History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download History as CSV",
        csv,
        "analysis_history.csv",
        "text/csv"
    )

st.divider()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<hr>
<div style="text-align:center;font-size:13px;color:gray;">
© 2026 TruthLens AI | Powered by Machine Learning | Developed by Avirup
</div>
""", unsafe_allow_html=True)