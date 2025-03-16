import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
import base64

# Initialize directories and files
os.makedirs("captured_images", exist_ok=True)
dataset_file = "user_input_data.csv"

# Preprocessing functions
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    return np.expand_dims(tf.keras.applications.resnet50.preprocess_input(img_array), axis=0)

def preprocess_tabular_features(features):
    max_values = [100, 200, 150, 180, 1, 1, 2, 2, 2, 5]
    return np.array(features) / max_values

# Load or train the model
try:
    model = load_model("heart_disease_prediction_model.keras")
    if model.input[1].shape[-1] != 10:
        raise ValueError("Model input shape mismatch. Retraining required.")
except (OSError, ValueError) as e:
    st.error(f"Model error: {e}. Retraining...")
    # Train the model (you can call your train_model() function here)
    model = load_model("heart_disease_prediction_model.keras")

# Add background image
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"Background image not found at: {image_file}")

# Streamlit UI
def main():
    # Set background image
    # Replace with the correct path to your background image
    background_image_path = r"background.jpeg"  # Use raw string or forward slashes
    add_bg_from_local(background_image_path)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Capture Images", "Enter Details", "Predict"])

    if page == "Home":
        st.title("Heart Disease Prediction")
        st.write("Welcome to the Heart Disease Prediction App!")
        st.write("Use the navigation menu on the left to capture images, enter your details, and predict your heart disease risk.")

    elif page == "Capture Images":
        st.title("Capture Images")
        st.write("### Step 1: Capture Images")
        capture_button = st.button("Capture Images")

        if "captured_images" not in st.session_state:
            st.session_state.captured_images = []

        if capture_button:
            cap = cv2.VideoCapture(0)
            for i in range(3):
                ret, frame = cap.read()
                if ret:
                    path = f"captured_images/image_{i + 1}.jpg"
                    cv2.imwrite(path, frame)
                    st.session_state.captured_images.append(preprocess_image(path))
                    st.image(frame, caption=f"Captured Image {i + 1}", use_column_width=True)
            cap.release()
            st.success("3 images captured successfully!")

    elif page == "Enter Details":
        st.title("Enter Your Details")
        st.write("### Step 2: Enter Your Details")

        if "user_features" not in st.session_state:
            st.session_state.user_features = []

        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        systolic_bp = st.number_input("Systolic BP", min_value=0, max_value=200, value=120)
        diastolic_bp = st.number_input("Diastolic BP", min_value=0, max_value=150, value=80)
        cholesterol = st.number_input("Cholesterol Level", min_value=0, max_value=180, value=150)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        family_history = st.selectbox("Family History", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        smoking_status = st.selectbox("Smoking Status", options=[0, 1, 2], format_func=lambda x: ["Non-smoker", "Smoker", "Ex-smoker"][x])
        physical_activity = st.selectbox("Physical Activity", options=[0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])
        medical_conditions = st.selectbox("Medical Conditions", options=[0, 1, 2], format_func=lambda x: ["None", "Diabetes", "Hypertension"][x])
        stress_level = st.slider("Stress Level", min_value=1, max_value=5, value=3)

        st.session_state.user_features = [age, systolic_bp, diastolic_bp, cholesterol, gender, family_history, smoking_status, physical_activity, medical_conditions, stress_level]

    elif page == "Predict":
        st.title("Predict Heart Disease Risk")
        st.write("### Step 3: Predict")

        if "captured_images" not in st.session_state or len(st.session_state.captured_images) == 0:
            st.error("Please capture images first!")
        elif "user_features" not in st.session_state or len(st.session_state.user_features) == 0:
            st.error("Please enter your details first!")
        else:
            try:
                # Prepare inputs
                tabular_features = preprocess_tabular_features([st.session_state.user_features])
                tabular_features_batch = np.repeat(tabular_features, len(st.session_state.captured_images), axis=0)
                user_image_batch = np.vstack(st.session_state.captured_images)

                # Make predictions
                predictions = model.predict([user_image_batch, tabular_features_batch])
                predicted_years = predictions[0].mean()
                has_heart_disease = int(predictions[1].mean() >= 0.5)
                risk_level = "High Risk" if predicted_years <= 5 else "Low Risk"

                # Display results
                st.write("### Prediction Results")
                st.write(f"Predicted years until heart disease: **{predicted_years:.2f} years**")
                st.write(f"Heart Disease Status: **{'Yes' if has_heart_disease else 'No'}**")
                st.write(f"Risk Level: **{risk_level}**")

                # Save user input to CSV
                save_user_input(st.session_state.user_features)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Function to save user input
def save_user_input(user_features):
    try:
        if os.path.exists(dataset_file):
            df = pd.read_csv(dataset_file)
        else:
            df = pd.DataFrame(columns=[
                "Age", "Systolic_BP", "Diastolic_BP", "Cholesterol_Level",
                "Gender", "Family_History", "Smoking_Status",
                "Physical_Activity", "Medical_Conditions", "Stress_Level"
            ])
        new_df = pd.DataFrame([user_features], columns=df.columns)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(dataset_file, index=False)
        st.success(f"User data saved to {dataset_file}.")
    except Exception as e:
        st.error(f"Error saving user data: {e}")

if __name__ == "__main__":
    main()