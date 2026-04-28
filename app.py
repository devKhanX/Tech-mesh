import streamlit as st
from PIL import Image
import base64
import numpy as np
import cv2
import os
from openai import OpenAI

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("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))

st.set_page_config(page_title="Multimodal AI Analyzer", layout="wide")

st.title("🧠 Multimodal Image Understanding & Storytelling")

# Sidebar Dashboard
st.sidebar.title("📁 Analysis Dashboard")

if "history" not in st.session_state:
    st.session_state.history = []

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])

# Tone selection
tone = st.selectbox("Select Story Tone", [
    "emotional", "funny", "dramatic", "formal", "kids", "detective"
])

# Image quality analysis
def analyze_quality(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    if brightness < 50:
        light = "Low light"
    elif brightness > 200:
        light = "Too bright"
    else:
        light = "Normal lighting"

    if blur < 100:
        sharpness = "Blurry"
    else:
        sharpness = "Clear"

    return f"{light}, {sharpness}"

# AI analysis
def analyze_image(image_bytes, tone):
    prompt = f"""
    Analyze this image and return clearly:

    Caption: One sentence
    Summary: 3-5 lines
    Objects: comma-separated list
    Emotion: mood or feeling
    Scene: type (indoor/outdoor/classroom/street/etc)
    Story: short creative story in {tone} tone
    Confidence: percentage (0-100)
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_bytes}"
                }}
            ]}
        ]
    )

    return response.choices[0].message.content


# MAIN PROCESS
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to base64
    image_bytes = base64.b64encode(uploaded_file.read()).decode()

    with st.spinner("🔍 Analyzing Image..."):
        result = analyze_image(image_bytes, tone)
        quality = analyze_quality(image)

    # Display results
    st.subheader("📊 AI Analysis")
    st.text(result)

    st.subheader("🖼 Image Quality")
    st.write(quality)

    # Save to dashboard
    st.session_state.history.append({
        "image": image,
        "result": result
    })

# Follow-up Q&A
st.subheader("❓ Ask a Follow-up Question")
question = st.text_input("Ask something about the image")

if question and "result" in locals():
    followup_prompt = f"""
    Based on this analysis:
    {result}

    Answer this:
    {question}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": followup_prompt}]
    )

    st.write(response.choices[0].message.content)

# Sidebar history
st.sidebar.subheader("Recent Analyses")

for item in st.session_state.history[-5:][::-1]:
    st.sidebar.image(item["image"], width=100)
    st.sidebar.write(item["result"][:80] + "...")
