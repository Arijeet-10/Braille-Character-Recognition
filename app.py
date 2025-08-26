import os
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image  # needed for img preprocessing

# Initialize Flask app
app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="quantized_pruned_model2_braille.tflite")
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels (A–Z)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z']

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_braille(img_path):
    # Load image as RGB (not grayscale)
    img = image.load_img(img_path, target_size=(28, 28), color_mode="rgb")
    img_array = image.img_to_array(img)  # shape (28, 28, 3)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0  # shape (1, 28, 28, 3)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction + confidence
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))

    return class_labels[predicted_class], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        prediction, confidence = predict_braille(filepath)

        return render_template(
            "index.html",
            prediction=f"Predicted: {prediction} (Confidence: {confidence:.2f})",
            image_path=filepath
        )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
