import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

from face_shape_predictor import miain, load_model
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------
MODEL_PATH = "model_85_nn_.pth"
TF_MODEL_PATH = "mobilenetv2_face_shape_finetuned_v6.keras"
HAIRSTYLE_PATH = "hairstyle_dataset_pro"

class_names_tf = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']


# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
@st.cache_resource
def load_face_shape_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_tf_model():
    return tf.keras.models.load_model(TF_MODEL_PATH)


pt_model = load_face_shape_model()
tf_model = load_tf_model()


# ---------------------------------------------------
# TF IMAGE PROCESSING (PIL ONLY)
# ---------------------------------------------------
def model2img(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_tf(img_path):
    img = model2img(img_path)
    pred = tf_model.predict(img)
    cls = np.argmax(pred, axis=1)[0]
    return class_names_tf[cls]


# ---------------------------------------------------
# GROQ LLM
# ---------------------------------------------------
load_dotenv("apiroute.env")

llm_high = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.6,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)


# ---------------------------------------------------
# HAIRSTYLE IMAGE DISPLAY
# ---------------------------------------------------
def hairstyle_image(gen, shape):
    base_path = os.path.join(HAIRSTYLE_PATH, "Men" if gen=="male" else "Women")
    shape_path = os.path.join(base_path, shape)

    if not os.path.exists(shape_path):
        st.error("No hairstyle folder found for this face shape.")
        return

    hairstyles = sorted([
        h for h in os.listdir(shape_path)
        if os.path.isdir(os.path.join(shape_path, h))
    ])

    st.subheader(f"📸 Hairstyle Visuals for {gen.capitalize()} – {shape}")

    for hairstyle in hairstyles:
        folder_path = os.path.join(shape_path, hairstyle)
        img_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(("jpg","jpeg","png","webp"))
        ][:3]

        st.markdown(f"### ✨ {hairstyle}")

        cols = st.columns(3)
        for i, img_path in enumerate(img_files):
            with cols[i]:
                st.image(img_path, use_container_width=True)


# ---------------------------------------------------
# LLM RECOMMENDATION CALL (UNTOUCHED PROMPT)
# ---------------------------------------------------
def get_llm_reco(shape, gender):
    user_prompt = f"""
You are a hairstyle and beard-style expert.
Use ONLY the hairstyle names and beard-style names given below.
Do NOT invent any new hairstyle or beard style names.
========================
HAIRSTYLE LIST (USE ONLY THESE)
========================
Male Hairstyles
- Round: Textured Quiff, Undercut with Longer Top, Side-Swept Medium Cut
- Square: Slick Back, Messy Textured Medium Cut, Side-Part Taper
- Oblong: Fringe Haircut, Side-Swept Medium Cut, Curly Top with Low Fade
- Heart: Medium Textured Crop, Wavy Side Part, Low-Height Pompadour
- Oval: Buzz Cut, Crew Cut, Medium Pompadour
Female Hairstyles
- Round: Long Layered Cut with Side Part, Wavy Lob Below Chin, Crown-Volume Layered Cut
- Square: Soft Curls with Layered Ends, Side-Swept Curtain Bangs, Shoulder-Length Waves
- Oblong: Curtain Bangs with Medium Waves, Chin-Length Bob, Shoulder-Length Blunt Cut
- Heart: Side-Swept Curls, Long Layers Below Jaw, Soft Waves with Curtain Bangs
- Oval: Long Straight Layers, Beach Waves, Blunt Bob
========================
BEARD LIST (USE ONLY THESE)
========================
- Round Face: Short Boxed Beard, Anchor Beard, Chin Strap with Mustache
- Square Face: Rounded Full Beard, Balbo Beard, Light Stubble with Fade
- Oblong Face: Full Beard, Chin Strap with Full Mustache, Short Boxed Beard
- Heart Face: Fuller Chin Beard, Light Stubble, Petite Goatee
- Oval Face: 3-Day Stubble, Even Full Beard, Goatee with Mustache
========================
TASK
========================
Detected face shape: {shape}
Detected gender: {gender}
Your job:
1. For the detected face shape, pick EXACTLY THREE hairstyles from the list
   for the detected gender.
2. For each chosen hairstyle:
   - Write the hairstyle name
   - Give a 3–4 line creative explanation of WHY it matches that face shape + gender.
3. If gender = "male":
   - Provide a section “Beard Styles”
   - Choose EXACTLY TWO beard styles from the matching list.
4. End with a short, elegant 2-line summary.
========================
STYLE OF WRITING
========================
- Highly creative
- Human-like
- Clear + confident
- No repetition
- No new hairstyle names
Begin now.
"""
    return llm_high.invoke(user_prompt).content


# ---------------------------------------------------
# COMBINED MODEL LOGIC
# ---------------------------------------------------
def combined_face_predict(img_path):
    pt_pred = list(miain(img_path, pt_model).keys())[0]
    tf_pred = predict_tf(img_path)
    return pt_pred, tf_pred


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("💇 AI Hairstyle & Beard Recommendation System")
st.write("Upload an image or take a live photo!")

tab1, tab2 = st.tabs(["Upload Image", "Live Camera"])

uploaded_img = None
with tab1:
    uploaded_img = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

with tab2:
    live_photo = st.camera_input("Take a live photo")

img_source = uploaded_img or live_photo

gender = st.selectbox("Select gender", ["male", "female"])


if img_source and st.button("Generate Recommendations"):
    img = Image.open(img_source).convert("RGB")
    img_path = "input_photo.png"
    img.save(img_path)

    st.image(img, caption="Input Image", use_container_width=True)

    # --------- COMBINED MODEL PREDICTION ----------
    pt_shape, tf_shape = combined_face_predict(img_path)

    st.info(f"**Face Expert Ethan Finch (Model 1):** {pt_shape}")
    st.info(f"**Face Expert Dr. Vanessa Smith (Model 2):** {tf_shape}")

    # SAME SHAPE → normal
    if pt_shape == tf_shape:
        st.success(f"Final Detected Face Shape: **{pt_shape}**")
        reco = get_llm_reco(pt_shape, gender)
        st.write(reco)
        hairstyle_image(gender, pt_shape)

    else:
        # DIFFERENT SHAPES → dual output
        st.warning("Both experts detected different face shapes. "
                   "This is possible if the image has mixed traits or low clarity.")

        # Ethan Finch section
        st.header(f"🔹 Expert Ethan Finch's View – {pt_shape}")
        reco1 = get_llm_reco(pt_shape, gender)
        st.write(reco1)
        hairstyle_image(gender, pt_shape)

        st.markdown("---")

        # Vanessa Smith section
        st.header(f"🔸 Expert Dr. Vanessa Smith's View – {tf_shape}")
        reco2 = get_llm_reco(tf_shape, gender)
        st.write(reco2)
        hairstyle_image(gender, tf_shape)
