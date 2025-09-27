import os
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np


# ========== TensorFlow Model Setup ==========
try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow is required. Run: pip install tensorflow")

MODEL_PATH = os.getenv("MODEL_PATH", "crop_disease_model.h5")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded!")

# ========== Class Labels ==========
CLASS_NAMES = [
    "Healthy", "Powdery Mildew", "Leaf Spot", "Fusarium Wilt", "Downy Mildew",
    "Bacterial Blight", "Early Blight", "Late Blight", "Rust", "Anthracnose", "Black Knot"
]

# ========== Flask App Setup ==========
app = Flask(__name__)
CORS(app)

# ========== Helper: Preprocess Image ==========
def preprocess_image(pil_img):
    img = pil_img.resize(model.input_shape[1:3])
    img_array = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ========== Route: Predict Crop Disease ==========
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        image = Image.open(BytesIO(file.read()))
        input_tensor = preprocess_image(image)
        predictions = model.predict(input_tensor)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        predicted_class = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class_{class_idx}"
        return jsonify({
            'prediction': predicted_class,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========== Route: Fertilizer Manual Recommendation ==========
OPTIMUM_NPK = {
    "rice": {'N': 90, 'P': 40, 'K': 40},
    "wheat": {'N': 120, 'P': 60, 'K': 40},
    "maize": {'N': 110, 'P': 40, 'K': 30},
    "cotton": {'N': 200, 'P': 60, 'K': 60},
    "sugarcane": {'N': 250, 'P': 100, 'K': 120},
}

def recommend_fertilizer(crop, n, p, k):
    crop = crop.lower()
    if crop not in OPTIMUM_NPK:
        return "⚠️ Crop not found in database."

    opt = OPTIMUM_NPK[crop]
    msg = []

    for nutrient, current, required in zip(['Nitrogen', 'Phosphorus', 'Potassium'], [n, p, k], [opt['N'], opt['P'], opt['K']]):
        diff = required - current
        if abs(diff) <= 5:
            msg.append(f"{nutrient} is optimal.")
        elif diff > 0:
            msg.append(f"Add {diff} kg/ha of {nutrient}.")
        else:
            msg.append(f"Reduce {abs(diff)} kg/ha of {nutrient}.")

    return " ".join(msg)

@app.route('/fertilizer', methods=['POST'])
def fertilizer():
    try:
        data = request.get_json()
        crop = data.get("crop", "").lower()
        n = float(data.get("nitrogen", 0))
        p = float(data.get("phosphorus", 0))
        k = float(data.get("potassium", 0))

        recommendation = recommend_fertilizer(crop, n, p, k)
        return jsonify({"recommendation": recommendation})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ========== Route: Fertilizer ML Prediction ==========
try:
    import pickle
    fertilizer_model = pickle.load(open("fertilizer_model.pkl", "rb"))
    print("✅ Fertilizer model loaded!")
except Exception as e:
    print("❌ Failed to load fertilizer model:", e)
    fertilizer_model = None

@app.route("/fertilizer-predict", methods=["POST"])
def predict_fertilizer():
    if fertilizer_model is None:
        return jsonify({"error": "Fertilizer model not available"}), 500

    data = request.get_json()

    try:
        input_values = [
            int(data['N']),
            int(data['P']),
            int(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['moisture']),
            int(data['soil_type']),
            int(data['crop_type'])
        ]
        prediction = fertilizer_model.predict([input_values])[0]
        fertilizer_names = ['Urea', 'Compost', 'DAP']
        result = fertilizer_names[prediction]
        return jsonify({"fertilizer": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ========== Run the App ==========
if __name__ == '__main__':
    app.run(debug=True, port=5000)
