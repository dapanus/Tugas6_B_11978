import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model(r"D:\semester 5\Mesin Learning\Tugas6_B_11978\model_mobilenet.h5")
class_name = ['Matang', 'Mentah']

# Fungsi untuk mengklasifikasi gambar
def classify_image(image_path):
    try:
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_name[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

# Fungsi untuk membuat progress bar khusus
def custom_progress_bar(confidence, color1, color2):
    percentage1 = confidence[0] * 100
    percentage2 = confidence[1] * 100
    progress_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;">
        <div style="width: {percentage1:.2f}%; background: {color1}; color: white; text-align: center; height: 24px; float: left;">
            {percentage1:.2f}%
        </div>
        <div style="width: {percentage2:.2f}%; background: {color2}; color: white; text-align: center; height: 24px; float: left;">
            {percentage2:.2f}%
        </div>
    </div>"""
    st.sidebar.markdown(progress_html, unsafe_allow_html=True)

# Tampilan utama aplikasi
st.title("Prediksi Kematangan Buah Naga - 11978")
uploaded_files = st.file_uploader("Unggah Gambar (Beberapa diperbolehkan)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            label, confidence = classify_image(uploaded_file.name)

            if label != "Error":
                primary_color = "#007BFF"
                secondary_color = "#FF4136"
                label_color = primary_color if label == "Matang" else secondary_color

                st.sidebar.write(f"Nama File: {uploaded_file.name}")
                st.sidebar.markdown(f"<h4 style='color: {label_color};'> Prediksi: {label} </h4>", unsafe_allow_html=True)

                st.sidebar.write("Confidence:")
                for i, cls_name in enumerate(class_name):
                    st.sidebar.write(f"- {cls_name}: {confidence[i] * 100:.2f}%")

                custom_progress_bar(confidence, primary_color, secondary_color)

                st.sidebar.write("---")
            else:
                st.sidebar.write(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
    else:
        st.sidebar.error("Silahkan unggah setidaknya satu gambar untuk diprediksi.")

if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)