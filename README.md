# YOLOv5 Object Detection API

This is a simple sample project that builds a RESTful API using Flask integrated with the YOLOv5 model for object detection on images. The project includes:

- **Server Side (Flask API):** Receives images via HTTP POST, runs YOLOv5 to detect objects, and returns the results in JSON format.
- **Client Side (Tkinter GUI):** A graphical interface that allows users to:
  - Upload an image.
  The interface sends data to the API, receives detection results, and displays the annotated image/video with bounding boxes and class names.

> **Note:**  
> This project is intended as a simple example for testing and internal development purposes. It does not use any authentication mechanisms (such as API keys). In production, you should add proper security measures such as HTTPS, API keys, rate limiting, etc.
> This app does not have object detection in videos and cameras because sending each frame or camera to the server for processing via HTTP can be slow and not achieve real-time performance if the computer or server is not powerful enough.
