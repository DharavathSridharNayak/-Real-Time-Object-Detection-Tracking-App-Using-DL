import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import tempfile
from PIL import Image
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
import requests

# Set page config
st.set_page_config(
    page_title="Real-Time Object Detection & Tracking",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful custom CSS for styling
st.markdown("""
    <style>
        :root {
            --primary-color: #6366f1;
            --secondary-color: #4f46e5;
            --background-color: #111827;
            --text-color: #f3f4f6;
            --card-color: #1f2937;
            --accent-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
        }
        
        body {
            color: var(--text-color);
            background-color: var(--background-color);
        }
        
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        .header {
            color: var(--primary-color);
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subheader {
            color: var(--primary-color);
            font-size: 1.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid var(--secondary-color);
            padding-bottom: 0.5rem;
        }
        .stButton>button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            border: none;
            transition: all 0.3s;
            font-weight: 500;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
        .css-1aumxhk {
            background-color: var(--card-color);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 4px solid var(--primary-color);
        }
        .model-card {
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            background-color: var(--card-color);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s;
            border-left: 4px solid var(--accent-color);
            height: 100%;
        }
        .model-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
            border-left: 4px solid var(--primary-color);
        }
        .model-card h3 {
            color: var(--primary-color);
            margin-top: 0;
        }
        .stSelectbox>div>div>select {
            border-radius: 8px;
            padding: 0.5rem;
            background-color: var(--card-color);
            color: var(--text-color);
            border: 1px solid var(--secondary-color);
        }
        .stSlider>div>div>div>div {
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        }
        .stTextInput>div>div>input {
            color: var(--text-color);
            background-color: var(--card-color);
            border-radius: 8px;
            border: 1px solid var(--secondary-color);
        }
        .stMarkdown {
            color: var(--text-color);
        }
        .stAlert {
            background-color: var(--card-color);
            color: var(--text-color);
            border-radius: 8px;
            border-left: 4px solid var(--accent-color);
        }
        .stProgress>div>div>div>div {
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        }
        .stCheckbox>label {
            color: var(--text-color);
        }
        .stRadio>label {
            color: var(--text-color);
        }
        .stFileUploader>label {
            color: var(--text-color);
        }
        .stMetric {
            color: var(--text-color);
            background-color: var(--card-color);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stInfo {
            background-color: rgba(59, 130, 246, 0.1);
            color: var(--text-color);
            border-left: 4px solid #3b82f6;
        }
        .stSuccess {
            background-color: rgba(16, 185, 129, 0.1);
            color: var(--text-color);
            border-left: 4px solid var(--accent-color);
        }
        .stWarning {
            background-color: rgba(245, 158, 11, 0.1);
            color: var(--text-color);
            border-left: 4px solid var(--warning-color);
        }
        .stError {
            background-color: rgba(239, 68, 68, 0.1);
            color: var(--text-color);
            border-left: 4px solid var(--error-color);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: var(--card-color);
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1.5rem;
            transition: all 0.3s;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color);
            color: white;
        }
        footer {
            color: #9ca3af;
            font-size: 0.9rem;
            text-align: center;
            padding: 1rem;
            margin-top: 2rem;
            border-top: 1px solid #374151;
        }
    </style>
""", unsafe_allow_html=True)

# App header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=100)
with col2:
    st.markdown('<div class="header">Real-Time Object Detection & Tracking</div>', unsafe_allow_html=True)
    st.markdown("Advanced computer vision for autonomous vehicles and surveillance systems")

# Navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Live Detection", "Video Processing", "Model Zoo", "Settings", "About"],
        icons=["house", "camera-video", "film", "boxes", "gear", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#1f2937"},
            "icon": {"color": "#6366f1", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#374151"},
            "nav-link-selected": {"background-color": "#6366f1"},
        }
    )

# Object detection class (simplified for demo)
class ObjectDetector:
    def __init__(self, model_type="yolov5"):
        self.model_type = model_type
        self.classes = ["person", "car", "truck", "bicycle", "motorcycle", "bus"]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
    def detect(self, image):
        # Convert image to numpy array
        frame = np.array(image)
        
        # Simulate detection by adding random bounding boxes
        height, width = frame.shape[:2]
        detections = []
        
        for _ in range(np.random.randint(2, 6)):
            class_id = np.random.randint(0, len(self.classes))
            confidence = np.random.uniform(0.7, 0.95)
            
            x = int(np.random.uniform(0, width * 0.8))
            y = int(np.random.uniform(0, height * 0.8))
            w = int(np.random.uniform(width * 0.1, width * 0.3))
            h = int(np.random.uniform(height * 0.1, height * 0.3))
            
            detections.append({
                "class_id": class_id,
                "confidence": confidence,
                "box": [x, y, x+w, y+h]
            })
        
        return detections
    
    def draw_detections(self, frame, detections):
        for detection in detections:
            class_id = detection["class_id"]
            confidence = detection["confidence"]
            box = detection["box"]
            
            color = self.colors[class_id]
            label = f"{self.classes[class_id]}: {confidence:.2f}"
            
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

# Video processor for WebRTC
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = ObjectDetector()
        self.confidence_threshold = 0.5
        self.tracking_enabled = True
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Perform detection
        detections = self.detector.detect(img)
        
        # Filter by confidence
        detections = [d for d in detections if d["confidence"] >= self.confidence_threshold]
        
        # Draw detections
        img = self.detector.draw_detections(img, detections)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Home Page
if selected == "Home":
    st.markdown('<div class="subheader">Welcome to Real-Time Object Detection</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            **Advanced computer vision system** for:
            - Autonomous vehicles 🚗
            - Surveillance systems 🏢
            - Traffic monitoring 🚦
            - Smart cities 🌆
            
            **Features:**
            - Real-time object detection
            - Multi-object tracking
            - Customizable models
            - High-performance inference
        """)
        
        st.button("Get Started →", key="home_get_started")
    
    with col2:
        st.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers-task-cv-object-detection.png", 
                 caption="Object Detection Example", use_column_width=True)
    
    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
        1. **Select a model** from our Model Zoo or upload your own
        2. **Choose input source** - live camera, video file, or image
        3. **Configure settings** - confidence threshold, tracking options
        4. **Run detection** and view real-time results
    """)

# Live Detection Page
elif selected == "Live Detection":
    st.markdown('<div class="subheader">Live Object Detection</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Webcam", "RTSP Stream"])
    
    with tab1:
        st.markdown("### Webcam Detection")
        st.info("This will use your device's camera for real-time object detection")
        
        # WebRTC configuration
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.05,
            help="Adjust the minimum confidence score for detections"
        )
        
        # Toggle tracking
        tracking_enabled = st.checkbox(
            "Enable Object Tracking", 
            value=True,
            help="Track objects across frames for consistent identification"
        )
        
        # Start WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
            webrtc_ctx.video_processor.tracking_enabled = tracking_enabled
    
    with tab2:
        st.markdown("### RTSP Stream Detection")
        st.warning("This feature requires an RTSP stream URL (e.g., from an IP camera)")
        
        rtsp_url = st.text_input("Enter RTSP Stream URL", "rtsp://example.com/stream")
        
        if st.button("Connect to Stream"):
            st.warning("RTSP stream processing would be implemented here in a production app")
            st.info(f"Would connect to: {rtsp_url}")

# Video Processing Page
elif selected == "Video Processing":
    st.markdown('<div class="subheader">Video File Processing</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a video file", 
        type=["mp4", "avi", "mov"],
        help="Upload a video file for object detection processing"
    )
    
    if uploaded_file is not None:
        st.success("Video file uploaded successfully!")
        
        # Save uploaded file to temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # Display video info
        video = cv2.VideoCapture(tfile.name)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Resolution", f"{width}x{height}")
        col2.metric("FPS", f"{fps:.2f}")
        col3.metric("Duration", f"{duration:.2f} seconds")
        
        # Processing options
        st.markdown("### Processing Options")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.05
        )
        
        tracking_enabled = st.checkbox("Enable Object Tracking", value=True)
        show_fps = st.checkbox("Show FPS Counter", value=True)
        
        # Process video button
        if st.button("Process Video"):
            st.warning("Video processing would be implemented here in a production app")
            
            # Simulate processing with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(1, 101):
                progress_bar.progress(i)
                status_text.text(f"Processing: {i}% complete")
                time.sleep(0.05)
            
            st.success("Video processing completed!")
            st.balloons()

# Model Zoo Page
elif selected == "Model Zoo":
    st.markdown('<div class="subheader">Model Selection</div>', unsafe_allow_html=True)
    
    st.markdown("""
        Choose from our pre-trained models or upload your own custom model.
        Different models offer different trade-offs between speed and accuracy.
    """)
    
    # Model cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="model-card">
                <h3>YOLOv5s</h3>
                <p><b>Type:</b> Object Detection</p>
                <p><b>Speed:</b> ⚡⚡⚡⚡⚡</p>
                <p><b>Accuracy:</b> ⭐⭐⭐</p>
                <p>Ultra-fast detection for real-time applications</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Select YOLOv5s", key="yolov5s"):
            st.session_state.selected_model = "yolov5s"
            st.success("YOLOv5s selected!")
    
    with col2:
        st.markdown("""
            <div class="model-card">
                <h3>Faster R-CNN</h3>
                <p><b>Type:</b> Object Detection</p>
                <p><b>Speed:</b> ⚡⚡</p>
                <p><b>Accuracy:</b> ⭐⭐⭐⭐⭐</p>
                <p>High accuracy for critical applications</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Select Faster R-CNN", key="frcnn"):
            st.session_state.selected_model = "faster_rcnn"
            st.success("Faster R-CNN selected!")
    
    with col3:
        st.markdown("""
            <div class="model-card">
                <h3>DeepSORT</h3>
                <p><b>Type:</b> Object Tracking</p>
                <p><b>Speed:</b> ⚡⚡⚡</p>
                <p><b>Accuracy:</b> ⭐⭐⭐⭐</p>
                <p>Tracking with deep learning features</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Select DeepSORT", key="deepsort"):
            st.session_state.selected_model = "deepsort"
            st.success("DeepSORT selected!")
    
    st.markdown("---")
    st.markdown("### Custom Model Upload")
    
    custom_model = st.file_uploader(
        "Upload your custom model (PyTorch or TensorFlow)", 
        type=["pt", "pth", "h5", "onnx"],
        help="Upload your custom trained model file"
    )
    
    if custom_model is not None:
        st.success("Custom model uploaded successfully!")
        st.info("Model would be loaded and validated here in a production app")

# Settings Page
elif selected == "Settings":
    st.markdown('<div class="subheader">Application Settings</div>', unsafe_allow_html=True)
    
    st.markdown("### Detection Parameters")
    confidence_threshold = st.slider(
        "Default Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05
    )
    
    iou_threshold = st.slider(
        "IOU Threshold (for NMS)", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.45, 
        step=0.05,
        help="Intersection over Union threshold for non-maximum suppression"
    )
    
    st.markdown("### Tracking Parameters")
    max_age = st.slider(
        "Max Track Age (frames)", 
        min_value=1, 
        max_value=100, 
        value=30,
        help="Number of frames to keep a track alive without detection"
    )
    
    min_hits = st.slider(
        "Min Detection Hits", 
        min_value=1, 
        max_value=10, 
        value=3,
        help="Number of detections needed before a track is confirmed"
    )
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

# About Page
elif selected == "About":
    st.markdown('<div class="subheader">About This Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
        **Real-Time Object Detection & Tracking System**
        
        This application demonstrates advanced computer vision capabilities for:
        - Autonomous vehicle perception systems
        - Surveillance and security applications
        - Traffic monitoring and analysis
        - Smart city infrastructure
        
        **Key Technologies:**
        - Deep learning-based object detection
        - Multi-object tracking algorithms
        - Real-time video processing
        - Edge computing optimization
        
        **Underlying Models:**
        - YOLOv5 for fast object detection
        - Faster R-CNN for high accuracy
        - DeepSORT for object tracking
        
        Developed with ❤️ using Streamlit and OpenCV.
    """)
    
    st.markdown("---")
    st.markdown("""
        **Disclaimer:** This is a demonstration application. For production use, 
        please ensure proper testing and validation of all components.
    """)

# Footer
st.markdown("""
    <footer>
        Real-Time Object Detection & Tracking | Powered by Streamlit and Hugging Face
    </footer>
""", unsafe_allow_html=True)
