import streamlit as st
import os
from PIL import Image
import easyocr
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# --- CONFIG ---
st.set_page_config(page_title="ReconMate – Multi-Receipt Recon", layout="centered")
st.title("🎬 ReconMate – Multi-Receipt Scanner & Recon")

# --- SESSION ---
if "data" not in st.session_state:
    st.session_state.data = []

# --- SIDEBAR ---
st.sidebar.header("🔐 API Setup")
provider = st.sidebar.selectbox("AI Provider", ["OpenAI", "OpenRouter"])
api_key = st.sidebar.text_input("API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4", "deepseek-chat", "claude-3-sonnet", "claude-3-opus"])

# --- MODEL MAP ---
model_map = {
    "gpt-4": "openai/gpt-4",
    "deepseek-chat": "deepseek-chat",
    "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
    "claude-3-opus": "anthropic/claude-3-opus-20240229",
}

if provider == "OpenRouter":
    base_url = "https://openrouter.ai/api/v1/chat/completions"
else:
    base_url = "https://api.openai.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload Image with Multiple Receipts", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🧾 Multi-Receipt Image", use_container_width=True)

    # --- OCR with Rotation ---
    with st.spinner("🔍 Running OCR on all angles..."):
        reader = easyocr.Reader(['en'], gpu=False)

        def get_text_rotations(image):
            rotations = [
                image,
                image.rotate(90, expand=True),
                image.rotate(180, expand=True),
                image.rotate(270, expand=True)
            ]
            texts = []
            for rot_img in rotations:
                img_np = np.array(rot_img)
                lines = reader.readtext(img_np, detail=0)
                texts.append("\n".join(lines))
            return "\n\n---\n\n".join(texts)

        full_text = get_text_rotations(img)

    st.subheader("🧾 Combined OCR Result (all angles)")
    st.text_area("OCR Text", full_text, height=250)

    # --- SPLIT RECEIPTS ---
    st.markdown("### 🧩 Split & Analyze Receipts")
    receipts = full_text.split("Total")
    receipts = [r.strip() + " Total" for r in receipts if r.strip()]
    st.info(f"🔍 Detected {len(receipts)} possible receipt(s)")

    structured_data = []

    for idx, receipt_text in enumerate(receipts):
        with st.spinner(f"Analyzing receipt {idx+1}..."):
            model_id = model_map.get(model)
            if not model_id:
                st.error("Model not supported.")
                st.stop()

            prompt = f"""
You are a film production finance assistant.

From the following receipt text, extract accurate structured data. Do not guess or invent fields — leave them blank if missing.

Return ONLY a JSON object with:
- vendor_name
- date
- total_amount
- tax_amount
- currency (e.g., ZAR, USD)
- payment_method (e.g., Credit Card, Cash, EFT)
- category: one of ["Food", "Transport", "Props", "Accommodation", "Fuel", "Other"]
- notes: short context for the expense

OCR TEXT:
\"\"\"
{receipt_text}
\"\"\"
"""

            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant for film producers."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }

            response = requests.post(base_url, headers=headers, json=payload)

            if response.status_code == 200:
                ai_reply = response.json()["choices"][0]["message"]["content"]
                try:
                    parsed = json.loads(ai_reply)
                    structured_data.append(parsed)
                except json.JSONDecodeError:
                    structured_data.append({
                        "vendor_name": "",
                        "date": "",
                        "total_amount": "",
                        "tax_amount": "",
                        "currency": "",
                        "payment_method": "",
                        "category": "",
                        "notes": f"⚠️ Could not parse JSON for receipt {idx+1}"
                    })
            else:
                structured_data.append({
                    "vendor_name": "",
                    "date": "",
                    "total_amount": "",
                    "tax_amount": "",
                    "currency": "",
                    "payment_method": "",
                    "category": "",
                    "notes": f"❌ API error {response.status_code}"
                })

    # --- DISPLAY RESULTS ---
    df = pd.DataFrame(structured_data)
    st.subheader("📊 Structured Recon Data")
    st.dataframe(df)

    # --- EXPORT TO GOOGLE DRIVE ---
    if st.button("📤 Export to Google Drive"):
        now = datetime.now()
        filename = f"ReconMate_Multi_Receipt_{now.strftime('%Y_%m')}.xlsx"
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
