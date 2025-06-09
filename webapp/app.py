import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

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
model_loaded = threading.Event()

def load_model_minimal():
    """Chargement ultra-optimisé du modèle avec gestion d'erreurs"""
    global model
    
    if model is None:
        with model_lock:
            if model is None:
                try:
                    print("🔄 Début chargement TensorFlow...")
                    

                    import tensorflow as tf
                    
                    tf.config.threading.set_inter_op_parallelism_threads(1)
                    tf.config.threading.set_intra_op_parallelism_threads(1)
                    
                    try:
                        tf.config.set_visible_devices([], 'GPU')
                    except:
                        pass
                    
                    print("🔄 Import Keras...")
                    from tensorflow.keras.models import load_model
                    
                    model_path = "model/mnist_model.h5"
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"❌ Modèle non trouvé: {model_path}")
                    
                    print("🔄 Chargement modèle...")
                    model = load_model(model_path, compile=False)
                    
                    gc.collect()
                    
                    print("✅ Modèle chargé avec succès")
                    model_loaded.set()
                    
                except Exception as e:
                    print(f"❌ Erreur chargement modèle: {str(e)}")
                    model = None
                    raise e
    
    return model

def preload_model():
    """Pré-charge le modèle en arrière-plan"""
    try:
        load_model_minimal()
    except Exception as e:
        print(f"⚠️ Pré-chargement échoué: {e}")

preload_thread = threading.Thread(target=preload_model, daemon=True)
preload_thread.start()

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
        
        try:
            header, encoded = image_data.split(",", 1)
        except ValueError:
            return jsonify({"error": "Format image invalide", "status": "error"}), 400
        
        try:
            img_bytes = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(img_bytes))
            img_array = np.array(img.convert("L").resize((28, 28)), dtype=np.float32)
            img_array = (img_array / 255.0).reshape(1, 784)
        except Exception as e:
            return jsonify({"error": "Erreur traitement image", "status": "error"}), 400
        
        del img_bytes, encoded, img
        
        try:
            current_model = load_model_minimal()
            if current_model is None:
                raise Exception("Modèle non disponible")
        except Exception as e:
            gc.collect()
            return jsonify({"error": "Modèle indisponible", "status": "error"}), 503
        
        try:
            pred = current_model.predict(img_array, verbose=0, batch_size=1)
            digit = int(np.argmax(pred))
            confidence = float(np.max(pred))
        except Exception as e:
            gc.collect()
            return jsonify({"error": "Erreur prédiction", "status": "error"}), 500
        
        del img_array, pred
        gc.collect()
        
        return jsonify({
            "digit": digit,
            "confidence": confidence,
            "status": "success"
        })
        
    except Exception as e:
        print(f"❌ Erreur predict: {str(e)}")
        gc.collect()
        return jsonify({"error": "Erreur serveur", "status": "error"}), 500

@app.route("/health")
def health():
    """Endpoint de santé avec info modèle"""
    model_status = "loaded" if model is not None else "not_loaded"
    return jsonify({
        "status": "ok", 
        "model_status": model_status,
        "ram_optimized": True
    })

@app.route("/model-status")
def model_status():
    """Status détaillé du modèle"""
    return jsonify({
        "loaded": model is not None,
        "loading": not model_loaded.is_set() if model is None else False
    })

@app.after_request
def cleanup(response):
    """Nettoyage automatique après chaque requête"""
    gc.collect()
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)