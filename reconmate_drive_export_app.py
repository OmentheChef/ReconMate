import streamlit as st
from PIL import Image
import pandas as pd
import os
import requests
import easyocr
from datetime import datetime
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# --- CONFIG ---
st.set_page_config(page_title="ReconMate – Film Recons Made Easy", layout="centered")
st.title("🎬 ReconMate – Film Recons Made Easy")

# --- SESSION DATA ---
if "data" not in st.session_state:
    st.session_state.data = []

# --- SIDEBAR: API KEYS & SETTINGS ---
st.sidebar.header("🔐 AI & Export Settings")
provider = st.sidebar.selectbox("Choose LLM Provider", ["OpenAI", "OpenRouter"])
api_key = st.sidebar.text_input("API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4", "deepseek-chat", "claude-3-sonnet", "claude-3-opus"])

# Map models
if provider == "OpenRouter":
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model_map = {
        "gpt-4": "openai/gpt-4",
        "deepseek-chat": "deepseek-chat",
        "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
        "claude-3-opus": "anthropic/claude-3-opus-20240229",
    }
else:
    base_url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model_map = {
        "gpt-4": "gpt-4"
    }

# --- RECEIPT UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload Receipt", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🧾 Receipt Preview", use_container_width=True)

    with st.spinner("🔍 Extracting text using OCR..."):
        reader = easyocr.Reader(['en'], gpu=False)
        ocr_result = reader.readtext(img)
        ocr_text = "\n".join([line[1] for line in ocr_result])

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

        with st.spinner("💬 Talking to AI..."):
            response = requests.post(base_url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            ai_output = result["choices"][0]["message"]["content"]
            st.subheader("✅ AI Categorization")
            st.markdown(ai_output)

            # Save to session
            st.session_state.data.append({
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Vendor": uploaded_file.name,
                "AI Output": ai_output,
                "OCR": ocr_text
            })
        else:
            st.error("❌ API Error")
            st.json(response.json())

# --- DATA TABLE DISPLAY ---
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    st.subheader("📊 Session Receipt Log")
    st.dataframe(df)

    # --- EXPORT TO GOOGLE DRIVE ---
    if st.button("📤 Export Monthly Report to Google Drive"):
        now = datetime.now()
        filename = f"ReconMate_Report_{now.strftime('%Y_%m')}.xlsx"
        filepath = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)
        df.to_excel(filepath, index=False)

        try:
            gauth = GoogleAuth()
            gauth.LocalWebserverAuth()
            drive = GoogleDrive(gauth)

            file = drive.CreateFile({'title': filename})
            file.SetContentFile(filepath)
            file.Upload()
            st.success(f"✅ Uploaded '{filename}' to Google Drive!")
        except Exception as e:
            st.error(f"Google Drive upload failed: {e}")
