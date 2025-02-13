from flask import Flask, request, jsonify
import torch
import cv2 as cv
import numpy as np


def predict_coordinate() :
    if request.method == 'POST' :
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if 'image' not in request.files :
            return jsonify({"error" : "No image provided"}), 400
        image_file = request.files['image']
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
        model.to(device)
        model.eval()
        class_names = model.names

        def detect_objects(image) :
            return model(image)

        results = detect_objects(img)
        detections_list = []
        try :
            detection_data = results.pred[0].tolist()
        except AttributeError :
            try :
                detection_data = results.result[0].tolist()
            except Exception as e :
                return jsonify({"error" : f"Cannot process detection results: {e}"}), 500
        for result in detection_data :
            x1, y1, x2, y2, confidence, class_id = result
            class_name = class_names.get(int(class_id))
            detection_dict = {
                'x1' : int(x1),
                'y1' : int(y1),
                'x2' : int(x2),
                'y2' : int(y2),
                'confidence' : float(confidence),
                'class_id' : int(class_id),
                'class_name' : class_name
            }
            detections_list.append(detection_dict)

        return jsonify(detections_list), 200
    else :
        return jsonify({"error" : "Invalid request method"}), 405
