import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import tempfile

# Load YOLOv8 model
model = YOLO("./runs/weights/best.pt")

# Streamlit App Configuration
st.set_page_config(
    page_title="Railway Track Defect Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Settings")
st.sidebar.write("Customize your detection experience.")

app_mode = st.sidebar.selectbox("Choose the App Mode", ["Image Folder", "Live Webcam", "Upload Video"])

# Add speed optimization slider for video modes
if app_mode in ["Live Webcam", "Upload Video"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Performance")
    frame_skip = st.sidebar.slider("Fast-Forward (Frame Skip)", 1, 5, 2, help="Increase this to process fewer frames and speed up video playback.")

# App Header
st.title("🚂 Railway Track Defect Detection")
st.write(
    "Analyze railway track images or live video for potential defects using the YOLOv8 model. "
    "This app organizes results in an easy-to-read format."
)

if app_mode == "Image Folder":
    folder_path = st.sidebar.text_input("📂 Enter Folder Path:")
    
    if folder_path:
        if os.path.isdir(folder_path):
            st.success(f"📁 Found folder: `{folder_path}`")
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(("jpg", "jpeg", "png"))
            ]
    
            if image_files:
                st.info(f"🔍 Found {len(image_files)} image(s). Processing...")
                
                # Summary Data
                summary_data = []
                # Tabs for each image
                tabs = st.tabs(image_files)
    
                for idx, image_file in enumerate(image_files):
                    # Read Image
                    image_path = os.path.join(folder_path, image_file)
                    image = Image.open(image_path)
    
                    # Convert PIL Image to OpenCV format
                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
                    # Perform inference
                    results = model(image_bgr)
    
                    # Annotate Image
                    annotated_image = results[0].plot()
                    
                    # Streamlit expects RGB, but YOLO/OpenCV outputs BGR. We convert it here so colors look natural!
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
                    # Detection Details
                    detections = []
                    for detection in results[0].boxes:
                        label = int(detection.cls)
                        confidence = float(detection.conf)
                        detections.append((label, confidence))
                    detection_status = "Defects Detected" if detections else "No Defects"
    
                    # Add to summary
                    summary_data.append({
                        "Image Name": image_file,
                        "Status": detection_status,
                        "Detections": ", ".join(
                            [f"Label: {d[0]}, Confidence: {d[1]:.2f}" for d in detections]
                        )
                    })
    
                    # Display in tab
                    with tabs[idx]:
                        st.subheader(f"Image: {image_file}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original Image", use_column_width=True)
                        with col2:
                            st.image(annotated_image, caption="Detected Image", use_column_width=True)
                        if detections:
                            st.markdown("### Detection Details")
                            for label, confidence in detections:
                                st.write(f"- **Label**: {label}, **Confidence**: {confidence:.2f}")
    
                # Display Summary Table
                st.markdown("## Detection Summary")
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.warning("⚠️ No valid images found in the folder.")
        else:
            st.error("🚫 The specified folder does not exist.")
    else:
        st.info("📝 Please enter a folder path to get started.")

elif app_mode == "Live Webcam":
    st.markdown("### 📷 Live Webcam Inference")
    st.write("Check the box below to start the webcam. Uncheck it to stop.")
    
    run_webcam = st.checkbox("Start Webcam")
    
    if run_webcam:
        # Create an empty placeholder to stream the video frames
        FRAME_WINDOW = st.image([])
        
        # 0 is the default built-in webcam
        cap = cv2.VideoCapture(0)
        
        frame_count = 0
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video from webcam. Is it connected and not being used by another app?")
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            # Perform faster inference by reducing resolution natively in YOLO
            results = model(frame, imgsz=480)
            
            # Draw bounding boxes and labels
            annotated_frame = results[0].plot()
            
            # Convert colors back to RGB so it looks correct in the browser
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update the image placeholder with the new frame
            FRAME_WINDOW.image(annotated_frame)
            
        cap.release()

elif app_mode == "Upload Video":
    st.markdown("### 🎥 Video File Inference")
    st.write("Upload a video file to run YOLO inference frame-by-frame.")
    
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Read the video using OpenCV
        cap = cv2.VideoCapture(tfile.name)
        FRAME_WINDOW = st.image([])
        
        # Create a cancel button to stop video processing early
        stop_video = st.button("Stop Processing")
        
        frame_count = 0
        while cap.isOpened() and not stop_video:
            ret, frame = cap.read()
            if not ret:
                st.success("Video processing complete.")
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            # Perform faster inference by reducing resolution natively in YOLO
            results = model(frame, imgsz=480)
            
            # Draw bounding boxes and labels
            annotated_frame = results[0].plot()
            
            # Convert colors back to RGB so it looks correct in the browser
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update the image placeholder with the new frame
            FRAME_WINDOW.image(annotated_frame)
            
        cap.release()
