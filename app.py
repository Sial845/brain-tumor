import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageEnhance
import csv
import os
import time

# --- Streamlit Config ---
st.set_page_config(
    page_title="Brain Tumor Classifier + Grad-CAM",
    page_icon="🧠",
    layout="wide"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .stButton>button { margin-top: 0.5rem; }
    .container { padding: 2rem 3rem; }
    .title { font-size: 2.5rem; color: #003366; font-weight: 700; text-align: center; margin-bottom: 1rem; }
    .subtitle { font-size: 1.25rem; color: #555; font-weight: 500; margin-bottom: 2rem; text-align: center; }
    .fixed-img-size { width: 224px !important; height: 224px !important; object-fit: contain; border-radius: 8px; border: 1px solid #ddd; }
    .footer-custom { position: fixed; left: 0; bottom: 0; width: 100%; background-color: #e6f7f1; color: #333; text-align: center; padding: 10px; font-size: 14px; border-top: 1px solid #ccc; z-index: 9999; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_models():
    return load_model("Proposed.h5", compile=False)

model = load_models()
class_labels = ['Meningioma', 'Pituitary', 'Glioma', 'Normal']


# --- Utilities ---
def preprocess_image(img, contrast_factor=1.0, zoom_factor=1.0, crop_area=None):
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    width, height = img.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
    if crop_area:
        img = img.crop(crop_area)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# ✅ Multi-output prediction FIX
def predict_class(img_array):
    global_pred, region_pred, combined_pred = model.predict(img_array, verbose=0)
    preds = combined_pred[0]
    class_index = np.argmax(preds)
    return class_labels[class_index], preds[class_index], preds

# ✅ Grad-CAM FIX for multi-output model
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block_16_project", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output[-1]]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap_v2(original, heatmap, save_img_path="gradcam.png", colormap=cv2.COLORMAP_JET):
    img = np.array(original.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, colormap)
    superimposed_img = heatmap_colored * 0.4 + img
    cv2.imwrite(save_img_path, superimposed_img)
    return superimposed_img

def log_to_csv(label, confidence, probs):
    path = "predictions_log.csv"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [timestamp, label, round(confidence, 4)] + [round(p, 4) for p in probs]
    header = ["Timestamp", "Label", "Confidence"] + [f"Class_{i}_Prob" for i in range(len(probs))]
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

# --- UI ---
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown('<h1 class="title">🧠 Brain Tumor Classifier + Grad-CAM</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">🔬 Classify Brain Tumor Types from MRI Images</h3>', unsafe_allow_html=True)

# --- Sidebar Tools ---
st.sidebar.title("🔧 Tools")
contrast_factor = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0)
zoom_factor = st.sidebar.slider("Zoom", 1.0, 2.0, 1.0)
crop_str = st.sidebar.text_input("Crop Area (left, top, right, bottom)", "0,0,224,224")
crop_area = tuple(map(int, crop_str.split(","))) if crop_str else None
colormap_option = st.sidebar.selectbox("Grad-CAM Color Map", ["JET", "HOT", "COOL", "VIRIDIS"])
colormap_dict = {"JET": cv2.COLORMAP_JET, "HOT": cv2.COLORMAP_HOT, "COOL": cv2.COLORMAP_COOL, "VIRIDIS": cv2.COLORMAP_VIRIDIS}
colormap = colormap_dict.get(colormap_option, cv2.COLORMAP_JET)

# --- File Upload ---
uploaded_files = st.file_uploader(
    "📄 Upload Brain MRI Image(s) (single or multiple)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 1:
    image_pil = PILImage.open(uploaded_files[0]).convert("RGB")
    tabs = st.tabs(["📸 Uploaded Image", "📈 Results", "🔥 Grad-CAM", "📝 Explanation", "⬇️ Download"])

    with tabs[0]:
        st.image(image_pil, caption="Uploaded Brain MRI Image", width=224)

    with st.spinner("🔍 Predicting... Please wait..."):
        start_time = time.time()
        img_array = preprocess_image(image_pil, contrast_factor, zoom_factor, crop_area)
        label, confidence, probs = predict_class(img_array)
        heatmap = make_gradcam_heatmap(img_array, model)
        overlay_heatmap_v2(image_pil, heatmap, "gradcam.png", colormap)
        log_to_csv(label, confidence, probs)
        prediction_time = time.time() - start_time

    with tabs[1]:
        st.success(f"✅ **Prediction:** {label} ({confidence*100:.2f}%)")
        st.info(f"⏱️ **Time:** {prediction_time:.2f} seconds")
        st.markdown("#### 🔢 Class Probabilities:")
        for i, cls in enumerate(class_labels):
            st.write(f"• {cls}: `{probs[i]:.4f}`")

    with tabs[2]:
        st.markdown("#### 🔍 Grad-CAM Heatmap")
        grad_img_pil = PILImage.open("gradcam.png")
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(image_pil)
        ax[0].axis("off")
        ax[0].set_title("Original")
        ax[1].imshow(grad_img_pil)
        ax[1].axis("off")
        ax[1].set_title("Grad-CAM")
        st.pyplot(fig)

    with tabs[3]:
        st.markdown("### ℹ️ Explanation")
        st.write("""
        This model classifies brain tumor types (Meningioma, Pituitary, Glioma, Normal).
        Grad-CAM heatmaps highlight the regions of the MRI scan that influenced the model’s decision.
        """)

    with tabs[4]:
        st.markdown("### ⬇️ Download")
        with open("gradcam.png", "rb") as f:
            st.download_button("Download Grad-CAM Image", f, file_name="gradcam.png", mime="image/png")

elif uploaded_files and len(uploaded_files) > 1:
    st.markdown("### 📊 Batch Prediction with Optional Labels")
    images = []
    filenames = []
    cols_img = st.columns(min(6, len(uploaded_files)))
    for idx, f in enumerate(uploaded_files):
        img = PILImage.open(f).convert("RGB")
        images.append(img)
        filenames.append(f.name)
        cols_img[idx % 6].image(img, caption=f.name, width=120)

    st.markdown("---")
    st.markdown("#### 🧑‍⚕️ Enter Ground Truth Labels (Optional):")
    user_labels = []
    cols_input = st.columns(min(4, len(images)))
    for idx in range(len(images)):
        label_input = cols_input[idx % 4].text_input(f"Label for {filenames[idx]}", key=f"user_label_{idx}")
        user_labels.append(label_input.strip())

    if st.button("Predict All and Save CSV"):
        model_preds = []
        for img in images:
            img_array = preprocess_image(img, contrast_factor, zoom_factor, crop_area)
            pred_label, _, _ = predict_class(img_array)
            model_preds.append(pred_label)

        import pandas as pd
        df = pd.DataFrame({
            "Image ID": filenames,
            "Ground Truth": user_labels,
            "Model Prediction": model_preds
        })
        st.dataframe(df)

        csv_file = "braintumor_predictions.csv"
        write_header = not os.path.exists(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(["Image ID", "Ground Truth", "Model Prediction"])
            for i in range(len(filenames)):
                writer.writerow([filenames[i], user_labels[i], model_preds[i]])

        st.success(f"✅ Predictions saved to `{csv_file}`")

else:
    st.info("Please upload one or more brain MRI images to begin.")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="footer-custom">© BrainTumorNet - Developed by Dr. Sohaib Asif</div>', unsafe_allow_html=True)