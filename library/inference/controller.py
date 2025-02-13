from flask import Blueprint
from API_With_yolov5.library.inference.services import predict_coordinate


app_reference = Blueprint('inference', __name__)


@app_reference.route('/v1/object-detection/yolov5', methods=['POST'])
def predict() :
    return predict_coordinate()
