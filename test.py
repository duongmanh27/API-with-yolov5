from flask import Flask, request, jsonify
from PIL import Image
import torch
import cv2 as cv
import numpy as np


app = Flask(__name__)


@app.route('/v1/object-detection/yolov5', methods=['POST'])
def predict() :
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

        def detect_objects(image) :
            return model(image)

        results = detect_objects(img)
        detections_list = []
        for result in results.pred[0].tolist() :
            x1, y1, x2, y2, confidence, class_id = map(int, result)
            detection_dict = {
                'x1' : int(x1),
                'y1' : int(y1),
                'x2' : int(x2),
                'y2' : int(y2),
                'confidence' : float(confidence),
                'class' : int(class_id)
            }
            detections_list.append(detection_dict)

        return jsonify(detections_list), 200
    else :
        return jsonify({"error" : "Invalid request method"}), 405


if __name__ == '__main__' :
    app.run(host='0.0.0.0', port=5000, debug=True)
