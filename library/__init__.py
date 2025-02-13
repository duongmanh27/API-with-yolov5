from flask import Flask
from API_With_yolov5.library.inference.controller import app_reference


def create_app() :
    app = Flask(__name__)
    app.register_blueprint(app_reference)
    return app
