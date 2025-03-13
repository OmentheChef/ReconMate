import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import os
from datetime import datetime
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import requests

# --- APP CONFIG ---
st.set_page_config(page_title="ReconMate", layout="centered")
st.title("🎬 ReconMate – Film Recons Made Easy")

# --- SESSION STATE INIT ---
if "data" not in st.session_state:
    st.session_state.data = []

# --- SIDEBAR: API KEYS & SETTINGS ---
st.sidebar.header("🔐 API Setup")
provider = st.sidebar.selectbox("Choose LLM Provider", ["OpenAI", "OpenRouter"])
api_key = st.sidebar.text_input("API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4", "deepseek-chat", "claude-3-sonnet", "claude-3-opus"])

if provider == "OpenRouter":
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    model_map = {
        "gpt-4": "openai/gpt-4",
        "deepseek-chat": "deepseek-ai/deepseek-chat",
        "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
        "claude-3-opus": "anthropic/claude-3-opus-20240229"
    }
else:
    base_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    model_map = {
        "gpt-4": "gpt-4"
    }

# --- RECEIPT UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload Receipt", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🧾 Receipt Preview", use_column_width=True)

    with st.spinner("🔍 Extracting text..."):
        ocr_text = pytesseract.image_to_string(img)

    st.text_area("🧾 OCR Result", ocr_text, height=150)

    # --- AI ANALYSIS ---
    if st.button("🧠 Analyze with AI"):
        prompt = f"""
You are a film production assistant. A team member submitted this receipt. Your job is to extract and label the key info.

OCR TEXT:
\"\"\"
{ocr_text}
\"\"\"

Return:
- Vendor
- Date
- Total amount
- Category (Props, Food, Transport, etc.)
- Any notes or observations
"""

        payload = {
            "model": model_map[model],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for film producers managing receipts."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        with st.spinner("Talking to AI..."):
            response = requests.post(base_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            st.subheader("✅ AI Result")
            st.markdown(content)

            # Save record to session
            st.session_state.data.append({
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Vendor": uploaded_file.name,
                "AI Output": content,
                "OCR": ocr_text
            })
        else:
            st.error("❌ API Error")
            st.json(response.json())

# --- DATA TABLE DISPLAY ---
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    st.subheader("📊 Receipt Log (This Session)")
    st.dataframe(df)

    # --- EXPORT TO GOOGLE DRIVE ---
    if st.button("💾 Save & Upload Monthly Report to Google Drive"):
        now = datetime.now()
        filename = f"ReconMate_Report_{now.strftime('%Y_%m')}.xlsx"
        filepath = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)
        df.to_excel(filepath, index=False)

        try:
            gauth = GoogleAuth()
            gauth.LocalWebserverAuth()  # Opens browser for first-time auth
            drive = GoogleDrive(gauth)

            file = drive.CreateFile({'title': filename})
            file.SetContentFile(filepath)
            file.Upload()
            st.success(f"✅ Uploaded {filename} to Google Drive")
        except Exception as e:
            st.error(f"Upload failed: {e}")
