# Braille Character Recognition with CNN Optimization

This project implements a deep learning pipeline for Braille character recognition using Convolutional Neural Networks (CNNs), with model optimization techniques such as pruning and quantization for efficient deployment. The project includes a Jupyter notebook for model development and optimization, and a Flask web app for interactive predictions.

---

## Project Structure

```
app.py
optimizing-cnn-braille-character-detection.ipynb
quantized_pruned_model2_braille.tflite
Braille Dataset/
    a1.JPG0dim.jpg
    ...
static/
    uploads/
templates/
    index.html
```

---

## Main Components

- **optimizing-cnn-braille-character-detection.ipynb**  
  Jupyter notebook for data preprocessing, CNN training, pruning, quantization, and model evaluation.

- **app.py**  
  Flask web application for uploading images and predicting Braille characters using the optimized model.

- **Braille Dataset/**  
  Directory containing Braille character images for training and testing.

- **quantized_pruned_model2_braille.tflite**  
  The final quantized and pruned TensorFlow Lite model for deployment.

- **index.html**  
  HTML template for the web interface.

---

## Features

- **Data Preprocessing:**  
  Loads and normalizes Braille character images, encodes labels, and splits data into training and test sets.

- **CNN Model:**  
  Deep CNN architecture with batch normalization, dropout, and dense layers for robust feature extraction.

- **Model Optimization:**  
  - **Pruning:** Reduces model size by removing less important weights.
  - **Quantization:** Converts the model to TensorFlow Lite with reduced precision for efficient inference.

- **Evaluation:**  
  Tracks accuracy and loss during training and after optimization.

- **Web App:**  
  Upload an image and get the predicted Braille character and confidence score.

---

## How to Run

### 1. Train and Optimize the Model

Open and run all cells in optimizing-cnn-braille-character-detection.ipynb to:

- Preprocess data
- Train the CNN
- Apply pruning and quantization
- Export the optimized `.tflite` model

### 2. Start the Web Application

1. Ensure you have Python 3, Flask, TensorFlow, and required dependencies installed.
2. Place the optimized model (quantized_pruned_model2_braille.tflite) in the project root.
3. Run the Flask app:

    ```sh
    python app.py
    ```

4. Open your browser and go to `http://localhost:5000`.

### 3. Use the Web App

- Upload a Braille character image.
- View the predicted character and confidence.

---

## Requirements

- Python 3.7+
- TensorFlow
- tensorflow-model-optimization
- Flask
- OpenCV
- scikit-learn
- numpy, pandas, matplotlib, seaborn, PIL

Install requirements with:

```sh
pip install tensorflow tensorflow-model-optimization flask opencv-python scikit-learn numpy pandas matplotlib seaborn pillow
```

---

## Notes

- The dataset should be placed in the `Braille Dataset/` directory.
- The notebook demonstrates both Adam and RMSProp optimizers and compares their performance.
- The web app uses the quantized and pruned TFLite model for fast inference.

---

## References

- TensorFlow Model Optimization Toolkit: https://www.tensorflow.org/model_optimization
- Flask Documentation: https://flask.palletsprojects.com/
- optimizing-cnn-braille-character-detection.ipynb
- app.py
- index.html

---

## License

This project is for educational and research purposes. Please check dataset and dependency licenses before commercial use.