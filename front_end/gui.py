import tkinter as tk
from tkinter import filedialog
import cv2 as cv
from PIL import Image, ImageTk
import requests
import json


def upload_image() :
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image_files", "*.jpg;*.jpeg;*.png;*.gif;*bmp")]
    )
    if file_path :
        ori_image = cv.imread(file_path)
        if ori_image is None :
            print("Cant read image")
            return
        url = 'http://127.0.0.1:5000/v1/object-detection/yolov5'
        with open(file_path, 'rb') as f :
            files = {'image' : f}
            try :
                response = requests.post(url, files=files)
                if response.status_code == 200 :
                    detections_result = response.json()
                    detection_label.config(text="Result found :\n" + str(detections_result))
                else :
                    detection_label.config(text="Error: {}".format(response.status_code))
            except Exception as e :
                detection_label.config(text="Request error : {}".format(e))
        h, w = ori_image.shape[:2]
        target_size = (800, 600)
        scale_x = target_size[0] / w
        scale_y = target_size[1] / h
        resized_img = cv.resize(ori_image, target_size)
        for detection in detections_result :
            x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
            confidence = detection['confidence']
            class_id = detection['class_id']
            class_name = detection['class_name']
            x1_display = int(x1 * scale_x)
            y1_display = int(y1 * scale_y)
            x2_display = int(x2 * scale_x)
            y2_display = int(y2 * scale_y)
            cv.rectangle(resized_img, (x1_display, y1_display), (x2_display, y2_display), (0, 255, 0), 2)
            cv.putText(resized_img, class_name, (x1_display, max(y1_display - 10, 10)), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)
        image_rgb = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image_pil)

        image_label.config(image=photo)
        image_label.image = photo


root = tk.Tk()
root.title("App inference")
root.geometry("1080x720")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=50)

image_label = tk.Label(root)
image_label.pack(padx=10, pady=10)

detection_label = tk.Label(root, text="", font=("Arial", 20), wraplength=480, justify=tk.LEFT)
detection_label.pack(pady=5, padx=5)

root.mainloop()
