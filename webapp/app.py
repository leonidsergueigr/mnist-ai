from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
import os

app = Flask(__name__)

model_path = "model/mnist_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier modèle '{model_path}' n'existe pas. Assurez-vous d'avoir entraîné le modèle d'abord.")

print("Chargement du modèle...")
model = load_model(model_path)
print("✅ Modèle chargé avec succès!")

print(f"Forme d'entrée attendue par le modèle: {model.input_shape}")
print(f"Forme de sortie du modèle: {model.output_shape}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json['image']
        header, encoded = data.split(",", 1)
        
        img_bytes = base64.b64decode(encoded)
        
        img = Image.open(io.BytesIO(img_bytes))
        
        img = img.convert("L").resize((28, 28))
        
        img_array = np.array(img)
        
        img_array = img_array.astype('float32') / 255.0
        
        img_flattened = img_array.reshape(1, 28*28)
        
        print(f"Forme de l'image envoyée au modèle: {img_flattened.shape}")
        
        pred = model.predict(img_flattened, verbose=0)
        
        digit = int(np.argmax(pred))
        confidence = float(np.max(pred))
        
        print(f"Prédiction: {digit}, Confiance: {confidence:.4f}")
        
        return jsonify({
            "digit": digit, 
            "confidence": confidence,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route("/health")
def health():
    """Route de santé pour vérifier que l'API fonctionne"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_input_shape": str(model.input_shape),
        "model_output_shape": str(model.output_shape)
    })

if __name__ == "__main__":
    print("Démarrage de l'application Flask...")
    print("🚀 Application disponible sur: http://localhost:5000")
    print("🔍 Route de santé disponible sur: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)