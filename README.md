# 🧠 MNIST AI - Reconnaissance de Chiffres Manuscrits

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![Précision](https://img.shields.io/badge/Précision-98.5%25-brightgreen.svg)](#performance)

Un système de reconnaissance de chiffres manuscrits utilisant un **réseau de neurones dense profond** avec TensorFlow. Interface web interactive permettant de dessiner et reconnaître des chiffres en temps réel.

## 🚀 Fonctionnalités

### 🎯 **Modèle IA**
- **Réseau dense profond** optimisé pour MNIST
- **Techniques d'optimisation** avancées (BatchNorm, Dropout, Early Stopping)
- **Précision élevée** : ~98.5% sur le dataset de test
- **Prédiction rapide** avec calcul de confiance

### 🌐 **Interface Web**
- **Canvas interactif** pour dessiner les chiffres
- **Design responsive** adapté mobile/desktop
- **Prédiction temps réel** avec visualisation de confiance
- **Interface moderne** avec glassmorphisme et animations

### ⚡ **Performance**
- **Entraînement optimisé** avec callbacks intelligents
- **Validation croisée** pour éviter le surapprentissage
- **Sauvegarde automatique** des meilleurs poids

## 🧠 Architecture du Modèle

### 📐 **Topologie du Réseau**

```
Input Layer (784 neurones)
       ↓
Dense Layer (512 neurones) + ReLU + BatchNorm + Dropout(30%)
       ↓
Dense Layer (256 neurones) + ReLU + BatchNorm + Dropout(40%)
       ↓
Dense Layer (128 neurones) + ReLU + BatchNorm + Dropout(40%)
       ↓
Dense Layer (64 neurones) + ReLU + BatchNorm + Dropout(30%)
       ↓
Output Layer (10 neurones) + Softmax
```

### 🔧 **Spécifications Techniques**

| Paramètre | Valeur |
|-----------|---------|
| **Architecture** | Réseau Dense Profond (5 couches) |
| **Paramètres totaux** | ~500,000 |
| **Fonction d'activation** | ReLU (couches cachées), Softmax (sortie) |
| **Optimiseur** | Adam (lr=0.001, β₁=0.9, β₂=0.999) |
| **Fonction de perte** | Categorical Crossentropy |
| **Régularisation** | Dropout (30-40%) + Batch Normalization |

### 🎛️ **Techniques d'Optimisation**

- **Batch Normalization** : Stabilise l'entraînement et accélère la convergence
- **Dropout** : Prévient le surapprentissage (30-40% selon les couches)
- **Early Stopping** : Arrêt automatique basé sur la précision de validation
- **Learning Rate Decay** : Réduction automatique du taux d'apprentissage
- **Model Checkpointing** : Sauvegarde des meilleurs poids

## 📊 Performance

### 🎯 **Métriques Globales**

| Métrique | Dataset d'entraînement | Dataset de validation | Dataset de test |
|----------|----------------------|----------------------|-----------------|
| **Précision** | 99.2% | 98.7% | 98.5% |
| **Perte** | 0.025 | 0.041 | 0.047 |

### 📈 **Performance par Chiffre**

| Chiffre | Précision | Exemples | Erreurs |
|---------|-----------|----------|---------|
| 0 | 99.1% | 980 | 9 |
| 1 | 99.2% | 1135 | 9 |
| 2 | 97.8% | 1032 | 23 |
| 3 | 98.3% | 1010 | 17 |
| 4 | 98.6% | 982 | 14 |
| 5 | 97.9% | 892 | 19 |
| 6 | 98.7% | 958 | 12 |
| 7 | 98.4% | 1028 | 16 |
| 8 | 97.5% | 974 | 24 |
| 9 | 97.8% | 1009 | 22 |

### 📊 **Courbes d'Apprentissage**

L'entraînement converge généralement en 15-25 époques avec early stopping, montrant :
- **Convergence stable** sans oscillations
- **Pas de surapprentissage** grâce aux techniques de régularisation
- **Amélioration continue** de la précision de validation

## 🛠️ Installation

### 📋 **Prérequis**

```bash
Python 3.8+
pip (gestionnaire de paquets Python)
```

### 📦 **Installation des Dépendances**

```bash
# Cloner le repository
git clone https://github.com/leonidsergueigr/mnist-ai.git
cd mnist-ai

# Installer les dépendances
pip install -r requirements.txt

## 📈 Entraînement

### 🚀 **Lancement de l'Entraînement**

```bash
cd model
python train.py
```

### ⚙️ **Configuration Personnalisée**

```python
# Modifier les hyperparamètres dans train.py
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
```

### 📊 **Monitoring**

L'entraînement affiche en temps réel :
- **Précision** (entraînement et validation)
- **Perte** (entraînement et validation)
- **Learning rate** actuel
- **Temps par époque**

### 💾 **Sauvegarde**

Le modèle final est sauvegardé sous `mnist_model.h5` avec :
- **Architecture complète**
- **Poids optimisés**
- **Configuration de compilation**

## 🌐 Interface Web

### 🚀 **Lancement du Serveur**

```bash
cd webapp
python app.py
```

L'interface sera accessible sur `http://localhost:5000`

### 🎨 **Fonctionnalités Interface**

- **Canvas 400x400px** pour dessiner les chiffres
- **Dessin fluide** avec traits continus
- **Boutons intuitifs** : Effacer et Prédire
- **Résultats détaillés** : chiffre prédit + niveau de confiance
- **Design responsive** adapté tous écrans
- **Animations** et transitions fluides

### 📱 **Compatibilité**

- ✅ **Desktop** : Chrome, Firefox, Safari, Edge
- ✅ **Mobile** : iOS Safari, Android Chrome
- ✅ **Tablette** : Support tactile optimisé

## 📱 Utilisation

### 1️⃣ **Dessiner un Chiffre**
- Utilisez la souris ou le doigt pour dessiner dans le canvas noir
- Tracez un chiffre de 0 à 9 en blanc
- Le trait est fluide et continu

### 2️⃣ **Obtenir la Prédiction**
- Cliquez sur "🔍 Prédire"
- Le modèle analyse votre dessin
- Résultat affiché avec niveau de confiance

### 3️⃣ **Recommencer**
- Cliquez sur "🗑️ Effacer" pour nettoyer le canvas
- Dessinez un nouveau chiffre

### 💡 **Conseils pour de Meilleurs Résultats**

- **Centrez** le chiffre dans le canvas
- **Utilisez des traits épais** et bien contrastés
- **Dessinez clairement** sans ambiguïté
- **Évitez** les dessins trop petits ou trop grands

## 🔧 API

### 🌐 **Endpoints Disponibles**

#### `POST /predict`
Prédit un chiffre à partir d'une image

**Paramètres :**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Réponse :**
```json
{
  "status": "success",
  "digit": 7,
  "confidence": 0.9845,
  "probabilities": [0.001, 0.002, 0.001, 0.003, 0.001, 0.002, 0.005, 0.9845, 0.003, 0.001]
}
```

#### `GET /health`
Vérifie l'état de l'API

**Réponse :**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 📁 Structure du Projet

```
mnist-ai/
├── 📄 README.md                 # Documentation
├── 📄 requirements.txt          # Dépendances Python
├── 📁 model/                  
│   ├── 🐍 train.py           # Script d'entraînement
│   ├── 🧠 mnist_model.h5     # Modèle entraîné
├── 📁 webapp/                  
│   ├── 🐍 app.py               # Serveur Flask pour l'interface web
│   └── 📁 templates/           
│       └── 🌐 index.html       # Page HTML principale pour l'interface utilisateur
```

```

## ⚠️ Limitations et Avertissements

### 🎯 **Limitations du Modèle**
- **Dataset spécifique** : Optimisé pour MNIST (28x28 pixels, fond noir, chiffre blanc)
- **Style de dessin** : Fonctionne mieux avec des chiffres similaires au style MNIST
- **Résolution fixe** : Performance optimale sur images 28x28 pixels
- **Caractères uniquement** : Reconnaît seulement les chiffres 0-9

### ⚠️ **Avertissements d'Usage**
- **Pas de garantie** : Le modèle peut commettre des erreurs
- **Usage éducatif** : Conçu principalement pour l'apprentissage
- **Validation requise** : Vérifiez les résultats pour applications critiques
- **Biais possibles** : Performance variable selon le style d'écriture
