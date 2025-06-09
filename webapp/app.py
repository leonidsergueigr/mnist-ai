import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, request, jsonify
import numpy as np
import io
import base64
import gc
import threading
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

model = None
model_lock = threading.Lock()

def load_model_minimal():
    """Chargement ultra-optimisé du modèle"""
    global model
    if model is None:
        with model_lock:
            if model is None:
                # Import uniquement quand nécessaire
                from tensorflow.keras.models import load_model
                
                model_path = "model/mnist_model.h5"
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
                
                print("Chargement modèle...")
                model = load_model(model_path)
                
                gc.collect()
                print(f"✅ Modèle chargé - RAM optimisée")
    return model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Données manquantes", "status": "error"}), 400
        
        image_data = data['image']
        if len(image_data) > 500000:
            return jsonify({"error": "Image trop grosse", "status": "error"}), 400
        
        _, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img.convert("L").resize((28, 28)), dtype=np.float32)
        img_array = (img_array / 255.0).reshape(1, 784)
        
        del img_bytes, encoded, img
        
        current_model = load_model_minimal()
        pred = current_model.predict(img_array, verbose=0, batch_size=1)
        
        digit = int(np.argmax(pred))
        confidence = float(np.max(pred))
        
        del img_array, pred
        gc.collect()
        
        return jsonify({
            "digit": digit,
            "confidence": confidence,
            "status": "success"
        })
        
    except Exception as e:
        gc.collect() 
        return jsonify({"error": "Erreur traitement", "status": "error"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "ram_optimized": True})

@app.after_request
def cleanup(response):
    gc.collect()
    return response

if __name__ == "__main__":
    # Configuration minimaliste
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)