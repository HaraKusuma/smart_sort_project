
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('healthy_vs_rotten.h5')

classes = sorted(os.listdir('dataset/train'))  # this line should match your dataset folder

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_name = None
    if request.method == "POST":
        img_file = request.files["file"]
        image_name = img_file.filename
        img_path = os.path.join("static", image_name)
        img_file.save(img_path)

        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        result = classes[class_index]

    return render_template("index.html", result=result, image_name=image_name)

if __name__ == "__main__":
    app.run(debug=True)
