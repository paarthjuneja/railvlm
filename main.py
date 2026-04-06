import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import tempfile
import time
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
from dotenv import load_dotenv

import re
from fpdf import FPDF

from google import genai

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

if 'audit_log' not in st.session_state:
    st.session_state['audit_log'] = []
    
if 'vlm_reports_buffer' not in st.session_state:
    st.session_state['vlm_reports_buffer'] = []

def build_pdf_report(audit_log):
    """Dynamically strings together cached images and VLM strings into a PDF byte array."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 24)
    pdf.cell(0, 50, txt="Railway AI Automated Audit Report", ln=True, align='C')
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, txt=f"Total Identified Anomalies: {len(audit_log)}", ln=True, align='C')
    pdf.cell(0, 10, txt=f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    
    for idx, entry in enumerate(audit_log):
        pdf.add_page()
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 10, txt=f"Defect Logging Reference #{idx+1}", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 10, txt=f"Timestamp: {entry['timestamp']}", ln=True)
        
        img_pil = Image.fromarray(entry['image'])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
            img_pil.save(tf.name)
            pdf.image(tf.name, x=10, y=40, w=190)
            
        pdf.ln(140) 

        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 10, txt="AI Auditor Context:", ln=True)
        pdf.set_font("Helvetica", size=11)
        
        safe_text = re.sub(r'[^\x00-\x7F]+', '', entry['report'])
        safe_text = safe_text.replace('**', '').replace('*', '-')
        
        pdf.multi_cell(0, 8, txt=safe_text)
        
    return pdf.output() 


@st.cache_resource
def load_yolo_model():
    model = YOLO("best.pt")
    if 0 in model.names:
        model.names[0] = "Structural Defect"  
    return model

@st.cache_resource
def load_moondream_model():
    model_id = "vikhyatk/moondream2"
    vlm = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return vlm, tokenizer

def generate_vlm_report(image, detections, engine, api_key=None, local_vlm=None, local_tokenizer=None):
    if len(detections) == 0:
        return "No detections present for isolation."
        
    box = detections[0].xyxy[0].cpu().numpy()
    x_min, y_min, x_max, y_max = [int(b) for b in box]
    bbox_str = f"X:{x_min} to {x_max}, Y:{y_min} to {y_max}"
    
    prompt = (
        f"You are an expert railway track safety inspector. "
        f"A critical structural defect was detected at coordinates [{bbox_str}]. "
        f"Analyze this exact sector in the provided visual telemetry. Describe the physical nature of the crack or hardware anomaly. "
        f"Keep your answer very brief and to the point. Also keep in mind that it may be a false positive and there may be no crack altogether. "
        f"Conclude by definitively assigning the Priority Severity level (Low, Medium, High, or Critical)."
    )
    
    if engine == "Gemini 2.5 Flash (Cloud API)":
        if not api_key:
            return "Gemini API Key missing! Please enter it securely in the sidebar."
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image, prompt]
            )
            return response.text
        except Exception as e:
            return f"Cloud Infrastructure Error: {str(e)}"
            
    elif engine == "Moondream2 (Local Edge)":
        if not local_vlm or not local_tokenizer:
            return "Local Model Engine not injected properly."
        try:
            enc_image = local_vlm.encode_image(image)
            return local_vlm.answer_question(enc_image, prompt, local_tokenizer)
        except Exception as e:
            return f"Local Compute Error: {str(e)}"

model = load_yolo_model()

st.set_page_config(
    page_title="Railway Track Defect Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Settings")
st.sidebar.write("Customize your detection experience.")

app_mode = st.sidebar.selectbox("Choose the App Mode", ["Image Folder", "Single Image", "Live Webcam", "Upload Video"])

st.sidebar.markdown("---")
st.sidebar.markdown("### VLM Analysis")
enable_vlm = st.sidebar.checkbox("Enable AI Auditor", value=False)

vlm_engine = "Gemini 2.5 Flash (Cloud API)"
api_key_input = ""
local_vlm, local_tokenizer = None, None

if enable_vlm:
    vlm_engine = st.sidebar.radio("Selected AI Engine", ["Gemini 2.5 Flash (Cloud API)", "Moondream2 (Local Edge)"])
    if vlm_engine == "Gemini 2.5 Flash (Cloud API)":
        api_key_input = st.sidebar.text_input(
            "Gemini API Key", 
            type="password", 
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Get your free API key at aistudio.google.com"
        )
    elif vlm_engine == "Moondream2 (Local Edge)":
        st.sidebar.warning("Moondream may take slightly longer to load...")
        with st.spinner("Waking up Moondream2..."):
            local_vlm, local_tokenizer = load_moondream_model()

if app_mode in ["Live Webcam", "Upload Video"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Performance")
    frame_skip = st.sidebar.slider("Fast-Forward (Frame Skip)", 1, 5, 2, help="Increase this to process fewer frames speed up video playback.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Automated Reporting")
enable_pdf = st.sidebar.checkbox("Enable PDF Audit Generation", help="Buffers visual history into a PDF file")

if enable_pdf:
    if len(st.session_state['audit_log']) > 0:
        st.sidebar.success(f"{len(st.session_state['audit_log'])} logs safely buffered.")
        pdf_bytes = build_pdf_report(st.session_state['audit_log'])
        st.sidebar.download_button(
            label="Download Complete PDF Audit",
            data=bytes(pdf_bytes),
            file_name=f"Railway_Audit_{int(time.time())}.pdf",
            mime="application/pdf"
        )
        if st.sidebar.button("Clear Tracking Buffer"):
            st.session_state['audit_log'] = []
            st.session_state['vlm_reports_buffer'] = []
            st.rerun()
    else:
        st.sidebar.info("Awaiting structural defects... No data buffered yet.")


st.title("Railway Track Defect Detection")
st.write(
    "Analyze railway track images or live video for potential defects using the YOLOv8 model. "
    "This app organizes results in an easy-to-read format."
)

if app_mode == "Image Folder":
    folder_path = st.sidebar.text_input("Enter Folder Path:")
    if folder_path:
        if os.path.isdir(folder_path):
            st.success(f"Found folder: `{folder_path}`")
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(("jpg", "jpeg", "png"))]
    
            if image_files:
                st.info(f"Found {len(image_files)} image(s). Processing...")
                summary_data = []
                tabs = st.tabs(image_files)
    
                for idx, image_file in enumerate(image_files):
                    image_path = os.path.join(folder_path, image_file)
                    image = Image.open(image_path).convert("RGB")
                    image_np = np.array(image)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    results = model(image_bgr)
    
                    annotated_image = results[0].plot()
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
                    detections = results[0].boxes
                    detection_status = "Defects Detected" if len(detections) > 0 else "No Defects"
    
                    summary_data.append({
                        "Image Name": image_file,
                        "Status": detection_status,
                        "Detections": ", ".join([f"Class: {int(d.cls)}, Conf: {float(d.conf):.2f}" for d in detections])
                    })
    
                    with tabs[idx]:
                        st.subheader(f"{image_file}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original Asset", use_column_width=True)
                        with col2:
                            st.image(annotated_image_rgb, caption="YOLOv8 Detection", use_column_width=True)
                            
                        if len(detections) > 0:
                            report_text = "VLM Auditor module offline."
                            if enable_vlm:
                                with st.spinner(f"Engaging {vlm_engine}..."):
                                    report_text = generate_vlm_report(image, detections, vlm_engine, api_key_input, local_vlm, local_tokenizer)
                                st.info(f"**{vlm_engine} Audit:**\n\n{report_text}")
                                
                            if enable_pdf:
                                st.session_state['audit_log'].append({
                                    "image": annotated_image_rgb,
                                    "report": report_text,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                })
    
                st.markdown("## Global Detection Summary")
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.warning("No valid physical imagery found in the target directory.")
        else:
            st.error("Target directory does not exist.")
    else:
        st.info("Awaiting folder path input.")

elif app_mode == "Single Image":
    st.markdown("### Single Image Analysis")
    uploaded_file = st.file_uploader("Upload Telemetry Payload", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        with col1:
            st.image(image, caption="Raw Payload", use_column_width=True)
            
        with st.spinner("Processing YOLOv8 Neural Architecture..."):
            results = model(image_bgr)
            annotated_image = results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(annotated_image_rgb, caption="Spatial Bounding Map", use_column_width=True)
            
            detections = results[0].boxes
            if len(detections) > 0:
                st.warning(f"Anomalous Signature Localized ({len(detections)} instances)")
                report_text = "VLM module disabled."
                
                if enable_vlm:
                    with st.spinner(f"Establishing secure uplink to {vlm_engine}..."):
                        report_text = generate_vlm_report(image, detections, vlm_engine, api_key_input, local_vlm, local_tokenizer)
                        
                    st.markdown("---")
                    st.markdown(f"### {vlm_engine} Live Audit")
                    st.info(report_text)
                    
                if enable_pdf:
                    st.session_state['audit_log'].append({
                        "image": annotated_image_rgb,
                        "report": report_text,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
            else:
                st.success("Nominal status confirmed. Systems optimal.")

elif app_mode == "Live Webcam":
    st.markdown("### Live Webcam Inference")
    run_webcam = st.checkbox("Initialize Webcam Interface")
    
    if run_webcam:
        col1, col2 = st.columns([2, 1])
        with col1:
            FRAME_WINDOW = st.image([])
        with col2:
            vlm_placeholder = st.empty()
            if enable_vlm:
                vlm_placeholder.info(f"Armed sequence: {vlm_engine}. Awaiting visual defect signatures...")
            else:
                vlm_placeholder.info("Running raw YOLO telemetry without VLM Audit.")
                
        cap = cv2.VideoCapture(0)
        frame_count = 0
        audited_track_ids = set()
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Hardware pipeline failed.")
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            results = model.track(frame, imgsz=480, persist=True, verbose=False)
            detections = results[0].boxes
            
            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame_rgb)
            
            if len(st.session_state['vlm_reports_buffer']) > 0:
                vlm_placeholder.warning(st.session_state['vlm_reports_buffer'][-1])
            
            if len(detections) > 0 and detections.id is not None:
                track_ids = detections.id.int().cpu().tolist()
                
                for i, trk_id in enumerate(track_ids):
                    if trk_id not in audited_track_ids:
                        audited_track_ids.add(trk_id)
                        
                        if enable_vlm:
                            def fetch_bg(pil_img, dets_subset, eng, key, l_vlm, l_tok, t_id, rgb_frame, enable_p):
                                st.session_state['vlm_reports_buffer'].append(f"🔍 **Analyzing Structural Anomaly #{t_id}...**")
                                res = generate_vlm_report(pil_img, dets_subset, eng, key, l_vlm, l_tok)
                                final_text = f"**{eng} Audit (Anomaly #{t_id}):**\n\n{res}"
                                st.session_state['vlm_reports_buffer'].append(final_text)
                                
                                if enable_p:
                                    st.session_state['audit_log'].append({
                                        "image": rgb_frame,
                                        "report": final_text,
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                    })
                                    
                            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            t = threading.Thread(
                                target=fetch_bg, 
                                args=(pil_img, detections[i:i+1], vlm_engine, api_key_input, local_vlm, local_tokenizer, trk_id, annotated_frame_rgb, enable_pdf)
                            )
                            add_script_run_ctx(t)
                            t.start()
                        else:
                            sync_report = f"Anomaly #{trk_id} tracked dynamically by YOLO. VLM text disabled."
                            st.session_state['vlm_reports_buffer'].append(sync_report)
                            if enable_pdf:
                                st.session_state['audit_log'].append({
                                    "image": annotated_frame_rgb,
                                    "report": sync_report,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                })
        cap.release()

elif app_mode == "Upload Video":
    st.markdown("### Video File Inference")
    uploaded_file = st.file_uploader("Insert Media Node", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        col1, col2 = st.columns([2, 1])
        with col1:
            FRAME_WINDOW = st.image([])
        with col2:
            vlm_placeholder = st.empty()
            if enable_vlm:
                vlm_placeholder.info(f"Armed {vlm_engine}. Chronological stream active...")
            else:
                vlm_placeholder.info("Running raw YOLO telemetry without VLM Audit.")
                
        stop_video = st.button("Halt Execution")
        frame_count = 0
        audited_track_ids = set()
        
        while cap.isOpened() and not stop_video:
            ret, frame = cap.read()
            if not ret:
                st.success("End of file stream reached.")
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            
            results = model.track(frame, imgsz=480, persist=True, verbose=False)
            detections = results[0].boxes
            
            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame_rgb)
            
            if len(st.session_state['vlm_reports_buffer']) > 0:
                vlm_placeholder.warning(st.session_state['vlm_reports_buffer'][-1])
            
            if len(detections) > 0 and detections.id is not None:
                track_ids = detections.id.int().cpu().tolist()
                
                for i, trk_id in enumerate(track_ids):
                    if trk_id not in audited_track_ids:
                        audited_track_ids.add(trk_id)
                        
                        if enable_vlm:
                            def fetch_bg(pil_img, dets_subset, eng, key, l_vlm, l_tok, t_id, rgb_frame, enable_p):
                                st.session_state['vlm_reports_buffer'].append(f"**Analyzing Structural Anomaly #{t_id}...**")
                                res = generate_vlm_report(pil_img, dets_subset, eng, key, l_vlm, l_tok)
                                final_text = f"**{eng} Audit (Anomaly #{t_id}):**\n\n{res}"
                                st.session_state['vlm_reports_buffer'].append(final_text)
                                
                                if enable_p:
                                    st.session_state['audit_log'].append({
                                        "image": rgb_frame,
                                        "report": final_text,
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                    })
                                    
                            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            t = threading.Thread(
                                target=fetch_bg, 
                                args=(pil_img, detections[i:i+1], vlm_engine, api_key_input, local_vlm, local_tokenizer, trk_id, annotated_frame_rgb, enable_pdf)
                            )
                            add_script_run_ctx(t)
                            t.start()
                        else:
                            sync_report = f"Anomaly #{trk_id} tracked dynamically by YOLO. VLM text disabled."
                            st.session_state['vlm_reports_buffer'].append(sync_report)
                            if enable_pdf:
                                st.session_state['audit_log'].append({
                                    "image": annotated_frame_rgb,
                                    "report": sync_report,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                })
        cap.release()
