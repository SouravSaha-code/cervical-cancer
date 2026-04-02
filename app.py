from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import pickle
from PIL import Image

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore

from werkzeug.utils import secure_filename
import io
import base64

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
IMG_SIZE = (224, 224)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model paths
RESNET_MODEL_PATH = 'resnet152_cervical_cancer.keras'
VGG_MODEL_PATH = 'vgg16_cervical_cancer.keras'
ENSEMBLE_CONFIG_PATH = 'ensemble_config.pkl'

# Class information with descriptions
CLASS_INFO = {
    'im_Dyskeratotic': {
        'name': 'Dyskeratotic',
        'description': 'Abnormal keratinization of individual cells',
        'risk': 'Moderate to High',
        'color': '#FF6B6B'
    },
    'im_Koilocytotic': {
        'name': 'Koilocytotic',
        'description': 'Cells with perinuclear halo, often associated with HPV',
        'risk': 'High',
        'color': '#FFA07A'
    },
    'im_Metaplastic': {
        'name': 'Metaplastic',
        'description': 'Cells undergoing transformation',
        'risk': 'Low to Moderate',
        'color': '#FFD93D'
    },
    'im_Parabasal': {
        'name': 'Parabasal',
        'description': 'Small, round cells from basal layer',
        'risk': 'Low',
        'color': '#6BCB77'
    },
    'im_Superficial-Intermediate': {
        'name': 'Superficial-Intermediate',
        'description': 'Normal mature cells from surface layers',
        'risk': 'Very Low',
        'color': '#4D96FF'
    }
}

class WeightedEnsembleModel:
    """Ensemble model for predictions"""
    def __init__(self, model_paths, weights):
        self.models = []
        self.weights = weights
        
        print("Loading models...")
        for path in model_paths:
            model = load_model(path)
            self.models.append(model)
            print(f"✓ Loaded: {path}")
    
    def predict(self, x):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(x, verbose=0)
            predictions.append(pred * weight)
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred
    
    def predict_classes(self, x):
        predictions = self.predict(x)
        return np.argmax(predictions, axis=1)

# Global variables for models
ensemble_model = None
config = None

def load_ensemble():
    """Load ensemble model and configuration"""
    global ensemble_model, config
    
    try:
        with open(ENSEMBLE_CONFIG_PATH, 'rb') as f:
            config = pickle.load(f)
        
        ensemble_model = WeightedEnsembleModel(
            model_paths=config['model_paths'],
            weights=config['weights']
        )
        
        print("✅ Ensemble model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess image for model prediction"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_prediction_details(predictions, class_indices):
    """Get detailed prediction information"""
    class_names = list(class_indices.keys())
    pred_probs = predictions[0]
    
    # Get top prediction
    top_idx = np.argmax(pred_probs)
    top_class = class_names[top_idx]
    top_confidence = float(pred_probs[top_idx]) * 100
    
    # Get all predictions sorted by confidence
    all_predictions = []
    for idx, prob in enumerate(pred_probs):
        class_name = class_names[idx]
        confidence = float(prob) * 100
        
        all_predictions.append({
            'class': class_name,
            'class_display': CLASS_INFO[class_name]['name'],
            'confidence': round(confidence, 2),
            'description': CLASS_INFO[class_name]['description'],
            'risk': CLASS_INFO[class_name]['risk'],
            'color': CLASS_INFO[class_name]['color']
        })
    
    # Sort by confidence
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'predicted_class': top_class,
        'predicted_class_display': CLASS_INFO[top_class]['name'],
        'confidence': round(top_confidence, 2),
        'description': CLASS_INFO[top_class]['description'],
        'risk_level': CLASS_INFO[top_class]['risk'],
        'color': CLASS_INFO[top_class]['color'],
        'all_predictions': all_predictions
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html', class_info=CLASS_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if ensemble_model is None:
        return jsonify({'error': 'Model not loaded. Please restart the server.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess and predict
        img_array = preprocess_image(filepath)
        predictions = ensemble_model.predict(img_array)
        
        # Get detailed results
        result = get_prediction_details(predictions, config['class_indices'])
        
        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        result['image_data'] = f"data:image/jpeg;base64,{img_data}"
        result['filename'] = filename
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if ensemble_model is not None else 'unhealthy',
        'models_loaded': ensemble_model is not None,
        'tensorflow_version': tf.__version__
    }
    return jsonify(status)

if __name__ == '__main__':
    print("="*80)
    print("CERVICAL CANCER CELL CLASSIFICATION - FLASK APPLICATION")
    print("="*80)
    
    # Load models on startup
    if load_ensemble():
        print("\n🚀 Starting Flask server...")
        print("📍 Access the application at: http://localhost:5000")
        print("="*80)
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load models. Please ensure model files exist:")
        print(f"   - {RESNET_MODEL_PATH}")
        print(f"   - {VGG_MODEL_PATH}")
        print(f"   - {ENSEMBLE_CONFIG_PATH}")
        print("="*80)