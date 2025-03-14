import streamlit as st
import base64
import requests
import json
from datetime import datetime
import pandas as pd
<<<<<<< HEAD
import openpyxl  # ✅ Required for Excel export
=======
>>>>>>> 80cf0e5f (WIP: commit local changes before pull)
import easyocr
import numpy as np
from PIL import Image

# --- CONFIG ---
st.set_page_config(page_title="ReconMate Vision – Debug", layout="centered")
st.title("🔍 ReconMate Vision – Debug Mode")

if "data" not in st.session_state:
    st.session_state.data = []

# --- SIDEBAR ---
st.sidebar.header("🔐 API Setup")
<<<<<<< HEAD

# Select provider
provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "OpenRouter"])

# Models per provider
provider_models = {
    "OpenAI": {
        "gpt-4": "gpt-4-vision"
    },
    "OpenRouter": {
        "claude-3.5-sonnet": "anthropic/claude-3-sonnet-20240229",
        "deepseek-r1": "deepseek-ai/deepseek-chat"
    }
}

available_models = list(provider_models[provider].keys())
model = st.sidebar.selectbox("Model", available_models)

# API Key input
api_key = st.sidebar.text_input("API Key", type="password")
if not api_key:
    st.warning("⚠️ Please enter your API key to continue.")
    st.stop()

# Validate model
model_id = provider_models[provider].get(model)
if not model_id:
    st.error("❌ Invalid model selected.")
    st.stop()

# Set API endpoint
base_url = "https://api.openai.com/v1/chat/completions" if provider == "OpenAI" else "https://openrouter.ai/api/v1/chat/completions"
=======
provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "OpenRouter"])
api_key = st.sidebar.text_input("API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4", "claude-3.5-sonnet", "deepseek-r1"])

# --- Model Map (Fixed DeepSeek model) ---
model_map = {
    "gpt-4": "gpt-4-1106-vision-preview",  # ✅ New GPT-4 Vision model
    "claude-3.5-sonnet": "anthropic/claude-3-sonnet-20240229",
    "deepseek-r1": "deepseek-ai/deepseek-chat"  # ✅ Correct DeepSeek model ID
}

# --- API Key Check ---
if not api_key:
    st.warning("⚠️ Please enter your API key in the sidebar to continue.")
    st.stop()

model_id = model_map.get(model)
if not model_id:
    st.error("❌ Invalid model selection.")
    st.stop()

if provider == "OpenAI":
    base_url = "https://api.openai.com/v1/chat/completions"
else:
    base_url = "https://openrouter.ai/api/v1/chat/completions"

>>>>>>> 80cf0e5f (WIP: commit local changes before pull)
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload Receipt Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="🧾 Uploaded Receipt", use_container_width=True)
    file_bytes = uploaded_file.read()

<<<<<<< HEAD
    # === GPT-4 Vision (OpenAI) ===
    if provider == "OpenAI" and model == "gpt-4":
=======
    # === GPT-4 Vision ===
    if model == "gpt-4":
>>>>>>> 80cf0e5f (WIP: commit local changes before pull)
        st.info("🧠 Using GPT-4 Vision")
        image_b64 = base64.b64encode(file_bytes).decode("utf-8")

        vision_payload = {
<<<<<<< HEAD
            "model": model_id,
=======
            "model": "gpt-4-1106-vision-preview",
>>>>>>> 80cf0e5f (WIP: commit local changes before pull)
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract this receipt as JSON. Fields: vendor_name, date, total_amount, tax_amount, currency, payment_method, category (Food, Transport, etc), and notes."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }

<<<<<<< HEAD
        response = requests.post(base_url, headers=headers, json=vision_payload)
=======
        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers, json=vision_payload)
>>>>>>> 80cf0e5f (WIP: commit local changes before pull)

    # === Claude / DeepSeek with OCR ===
    else:
        st.info("🔍 Using OCR + Claude/DeepSeek")
        image = Image.open(uploaded_file)
        reader = easyocr.Reader(['en'], gpu=False)
        ocr_text = "\n".join(reader.readtext(np.array(image), detail=0))

        st.subheader("🧾 OCR Output")
        st.text_area("OCR Text", ocr_text, height=200)

        prompt = f"""
You are a film production finance assistant.

From the following receipt text, extract structured data in JSON format with these fields:
- vendor_name
- date
- total_amount
- tax_amount
- currency
- payment_method
- category (choose from: Food, Transport, Props, Accommodation, Fuel, Other)
- notes

Do not guess. Leave fields blank if missing.

Receipt Text:
\"\"\"
{ocr_text}
\"\"\"
"""

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for film producers."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }

        response = requests.post(base_url, headers=headers, json=payload)

<<<<<<< HEAD
    # === PARSE AI RESPONSE ===
=======
    # === PARSE RESPONSE ===
>>>>>>> 80cf0e5f (WIP: commit local changes before pull)
    if response.status_code == 200:
        ai_reply = response.json()["choices"][0]["message"]["content"]

        st.subheader("🧠 Raw AI Output")
        st.text_area("AI Response", ai_reply, height=200)

        try:
            parsed = json.loads(ai_reply)
            st.success("✅ JSON parsed successfully!")
            st.json(parsed)
            st.session_state.data.append(parsed)
        except Exception as e:
            st.error("❌ Could not parse AI response as JSON.")
            st.text(ai_reply)
    else:
        st.error("❌ API Error")
        try:
            st.json(response.json())
        except:
            st.write(response.text)

<<<<<<< HEAD
# === EXPORT + DISPLAY ===
=======
# === DISPLAY + EXPORT ===
>>>>>>> 80cf0e5f (WIP: commit local changes before pull)
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    st.subheader("📊 Recon Table")
    st.dataframe(df)

    if st.button("📥 Export to Excel"):
        filename = f"recon_debug_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        df.to_excel(filename, index=False)
        with open(filename, "rb") as f:
            st.download_button("Download Excel", f, file_name=filename)
