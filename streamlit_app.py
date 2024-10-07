import io
import time
from pathlib import Path

import cv2
import torch

from ultralytics.utils.checks import check_requirements


def inference(model=None):
  """Runs real-time object detection on video input using Ultralytics YOLOv8 in a Streamlit application."""
  check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
  import streamlit as st

  from ultralytics import YOLO
  import yaml  # Using standard yaml library

  # Hide main menu style
  menu_style_cfg = """"""

  # Main title of streamlit application
  main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; 
                           font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                  YOLO_Chilli Web Application
                  </h1></div>"""

  # Subtitle of streamlit application
  sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; 
                  font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                  Experience real-time chilli detection on your webcam or with video input. 🚀</h4>
                  </div>"""

  # Set html page configuration
  st.set_page_config(page_title="IIT-KGP-AGFE", layout="wide", initial_sidebar_state="auto")

  # Append the custom HTML
  st.markdown(menu_style_cfg, unsafe_allow_html=True)
  st.markdown(main_title_cfg, unsafe_allow_html=True)
  st.markdown(sub_title_cfg, unsafe_allow_html=True)

  # Add ultralytics logo in sidebar
  with st.sidebar:
      logo = "https://www.iitkgp.ac.in/assets/pages/images/logo.png"
      st.image(logo, width=250)

  # Add elements to vertical setting menu
  st.sidebar.title("User Configuration")

  # Add video source selection dropdown
  source = st.sidebar.selectbox(
      "Video",
      ("webcam", "video"),
  )

  vid_file_name = ""
  if source == "video":
      vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
      if vid_file is not None:
          g = io.BytesIO(vid_file.read())  # BytesIO Object
          vid_location = "ultralytics.mp4"
          with open(vid_location, "wb") as out:  # Open temporary file as bytes
              out.write(g.read())  # Read bytes into file
          vid_file_name = "ultralytics.mp4"
  elif source == "webcam":
      vid_file_name = 0

  # Add dropdown menu for model selection
  # Load models from the 'models' directory
  model_dir = Path('models')
  model_dir.mkdir(exist_ok=True)  # Create the models directory if it doesn't exist

  # List all .pt files in the models directory
  model_files = list(model_dir.glob('*.pt'))

  # Get the model names without the .pt extension for display
  available_models = [f.stem for f in model_files]

  # Check if available_models is empty
  if not available_models:
      st.error("No models available. Please check your model directory.")
      return  # Exit the function if no models are available

  selected_model = st.sidebar.selectbox("Model", available_models)

  # Check if selected_model is None or empty
  if selected_model is None or selected_model == "":
      st.error("Please select a model.")
      return  # Exit the function if no model is selected

  # Load the selected YOLO model
  with st.spinner("Loading model..."):
      model_path = model_dir / (selected_model + '.pt')
      if not model_path.exists():
          # Attempt to download the model if it does not exist locally
          # You can implement code to download from a custom source
          st.error(f"Model {selected_model} not found in the models directory.")
          return

      model = YOLO(str(model_path))  # Load the YOLO model

      # Load custom class names from data.yaml
      data_yaml_path = 'data.yaml'
      if not Path(data_yaml_path).exists():
          st.error(f"data.yaml file not found.")
          return

      # Use standard yaml library
      with open(data_yaml_path, 'r') as f:
          data_yaml = yaml.safe_load(f)
      class_names = data_yaml['names']

  # st.success("Model loaded successfully!")

  # Multiselect box with class names and get indices of selected classes
  selected_classes = st.sidebar.multiselect("Classes", class_names, default=class_names)
  selected_ind = [class_names.index(option) for option in selected_classes]

  if not isinstance(selected_ind, list):  # Ensure selected_ind is a list
      selected_ind = list(selected_ind)

  enable_trk = st.sidebar.radio("Enable Tracking", ("Yes", "No"))
  conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.60, 0.01))
  iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))

  col1, col2 = st.columns(2)
  org_frame = col1.empty()
  ann_frame = col2.empty()

  fps_display = st.sidebar.empty()  # Placeholder for FPS display

  if st.sidebar.button("Start"):
      videocapture = cv2.VideoCapture(vid_file_name)  # Capture the video

      if not videocapture.isOpened():
          st.error("Could not open webcam or video.")
          return

      stop_button = st.button("Stop")  # Button to stop the inference

      while videocapture.isOpened():
          success, frame = videocapture.read()
          if not success:
              st.warning("Failed to read frame from webcam/video.")
              break

          prev_time = time.time()

          # Store model predictions
          if enable_trk == "Yes":
              results = model.track(frame, conf=conf, iou=iou, classes=selected_ind, persist=True)
          else:
              results = model(frame, conf=conf, iou=iou, classes=selected_ind)
          annotated_frame = results[0].plot()  # Add annotations on frame

          # Calculate model FPS
          curr_time = time.time()
          fps = 1 / (curr_time - prev_time)
          prev_time = curr_time

          # display frame
          org_frame.image(frame, channels="BGR")
          ann_frame.image(annotated_frame, channels="BGR")

          if stop_button:
              videocapture.release()  # Release the capture
              torch.cuda.empty_cache()  # Clear CUDA memory
              st.stop()  # Stop streamlit app

          # Display FPS in sidebar
          fps_display.metric("FPS", f"{fps:.2f}")

      # Release the capture
      videocapture.release()

  # Clear CUDA memory
  torch.cuda.empty_cache()

  # Destroy window
  # cv2.destroyAllWindows()


# Main function call
if __name__ == "__main__":
  inference()
