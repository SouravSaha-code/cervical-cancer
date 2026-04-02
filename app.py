from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pickle
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing import image     # type: ignore
from werkzeug.utils import secure_filename
import base64
import gdown

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
MAX_FILE_SIZE      = 16 * 1024 * 1024
IMG_SIZE           = (224, 224)

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Google Drive File IDs ─────────────────────────────────────────────────────
DRIVE_FILES = {
    'resnet152_cervical_cancer.keras': '1Wf8F0CtgJWt2SiiaZTuxcoCNU0WM5oIe',
    'vgg16_cervical_cancer.keras':     '1QbqJeqeUQk35W_d7oiGNFWN35HiNHAUg',
    'ensemble_config.pkl':             '1keW_t8NMFbVO8Mzkljn3QrSzAhVTk8KH',
}

ENSEMBLE_CONFIG_PATH = 'ensemble_config.pkl'

# ── Class metadata ────────────────────────────────────────────────────────────
CLASS_INFO = {
    'im_Dyskeratotic': {
        'name': 'Dyskeratotic',
        'description': 'Abnormal keratinization of individual cells',
        'risk': 'Moderate to High',
        'color': '#FF6B6B',
    },
    'im_Koilocytotic': {
        'name': 'Koilocytotic',
        'description': 'Cells with perinuclear halo, often associated with HPV',
        'risk': 'High',
        'color': '#FFA07A',
    },
    'im_Metaplastic': {
        'name': 'Metaplastic',
        'description': 'Cells undergoing transformation',
        'risk': 'Low to Moderate',
        'color': '#FFD93D',
    },
    'im_Parabasal': {
        'name': 'Parabasal',
        'description': 'Small, round cells from basal layer',
        'risk': 'Low',
        'color': '#6BCB77',
    },
    'im_Superficial-Intermediate': {
        'name': 'Superficial-Intermediate',
        'description': 'Normal mature cells from surface layers',
        'risk': 'Very Low',
        'color': '#4D96FF',
    },
}

# ── Download models from Google Drive ────────────────────────────────────────
def download_models():
    """Download model files from Google Drive if not already present."""
    print("Checking model files...")
    for filename, file_id in DRIVE_FILES.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)
            print(f"  ✓ Downloaded: {filename}")
        else:
            print(f"  ✓ Already exists: {filename}")

# ── Ensemble model ────────────────────────────────────────────────────────────
class WeightedEnsembleModel:
    def __init__(self, model_paths, weights):
        self.models  = []
        self.weights = weights
        print("Loading models...")
        for path in model_paths:
            mdl = self._load_model(path)
            self.models.append(mdl)
            print(f"  ✓ Loaded: {path}")

    def _load_model(self, path):
        """Load model handling both old and new Keras formats."""
        import zipfile, json, h5py

        # Try standard load first
        try:
            import tf_keras
            return tf_keras.models.load_model(path, compile=False)
        except Exception:
            pass

        # Fallback: manual weight loading
        print(f"  Using manual weight loading for: {path}")
        with zipfile.ZipFile(path, 'r') as z:
            z.extractall(path + '_extracted')

        with open(path + '_extracted/config.json') as f:
            config = json.load(f)

        model = tf.keras.models.model_from_json(json.dumps(config))

        with h5py.File(path + '_extracted/model.weights.h5', 'r') as f:
            for layer in model.layers:
                key = f'_layer_checkpoint_dependencies\\{layer.name}'
                if key in f and len(layer.weights) > 0:
                    grp = f[key]
                    if 'vars' in grp:
                        weight_values = [
                            grp['vars'][k][()]
                            for k in sorted(grp['vars'].keys(), key=lambda x: int(x))
                        ]
                        try:
                            layer.set_weights(weight_values)
                        except Exception:
                            pass

        return model

    def predict(self, x):
        weighted = [m.predict(x, verbose=0) * w
                    for m, w in zip(self.models, self.weights)]
        return np.sum(weighted, axis=0)

    def predict_classes(self, x):
        return np.argmax(self.predict(x), axis=1)


# ── Global state ──────────────────────────────────────────────────────────────
ensemble_model = None
config         = None


def load_ensemble():
    global ensemble_model, config

    try:
        with open(ENSEMBLE_CONFIG_PATH, 'rb') as f:
            config = pickle.load(f)

        required = {'model_paths', 'weights', 'class_indices'}
        missing  = required - set(config.keys())
        if missing:
            print(f"❌ ensemble_config.pkl is missing keys: {missing}")
            return False

        ensemble_model = WeightedEnsembleModel(
            model_paths=config['model_paths'],
            weights=config['weights'],
        )
        print("✅ Ensemble model loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_mime_type(filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    return {
        'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
        'png': 'image/png',  'bmp':  'image/bmp',
        'tif': 'image/tiff', 'tiff': 'image/tiff',
    }.get(ext, 'image/jpeg')


def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def get_prediction_details(predictions, class_indices):
    class_names = list(class_indices.keys())
    pred_probs  = predictions[0]

    top_idx        = int(np.argmax(pred_probs))
    top_class      = class_names[top_idx]
    top_confidence = float(pred_probs[top_idx]) * 100

    all_predictions = []
    for idx, prob in enumerate(pred_probs):
        cname = class_names[idx]
        info  = CLASS_INFO.get(cname, {
            'name': cname, 'description': 'N/A',
            'risk': 'Unknown', 'color': '#9E9E9E',
        })
        all_predictions.append({
            'class':         cname,
            'class_display': info['name'],
            'confidence':    round(float(prob) * 100, 2),
            'description':   info['description'],
            'risk':          info['risk'],
            'color':         info['color'],
        })

    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

    top_info = CLASS_INFO.get(top_class, {
        'name': top_class, 'description': 'N/A',
        'risk': 'Unknown', 'color': '#9E9E9E',
    })

    return {
        'predicted_class':         top_class,
        'predicted_class_display': top_info['name'],
        'confidence':              round(top_confidence, 2),
        'description':             top_info['description'],
        'risk_level':              top_info['risk'],
        'color':                   top_info['color'],
        'all_predictions':         all_predictions,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html', class_info=CLASS_INFO)


@app.route('/predict', methods=['POST'])
def predict():
    if ensemble_model is None:
        return jsonify({'error': 'Model not loaded. Please try again later.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(sorted(ALLOWED_EXTENSIONS))}'
        }), 400

    filepath = None
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_array   = preprocess_image(filepath)
        predictions = ensemble_model.predict(img_array)
        result      = get_prediction_details(predictions, config['class_indices'])

        mime = get_mime_type(filename)
        with open(filepath, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')

        result['image_data'] = f"data:{mime};base64,{img_b64}"
        result['filename']   = filename

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


@app.route('/health')
def health():
    return jsonify({
        'status':             'healthy' if ensemble_model is not None else 'unhealthy',
        'models_loaded':      ensemble_model is not None,
        'tensorflow_version': tf.__version__,
    })


# ── Startup ───────────────────────────────────────────────────────────────────
# Download models then load them
download_models()

if load_ensemble():
    print("✅ App ready!")
else:
    print("⚠️ App started but models failed to load.")

if __name__ == '__main__':
    print("=" * 70)
    print("  CERVICAL CANCER CELL CLASSIFICATION — FLASK APPLICATION")
    print("=" * 70)
    print("📍 http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
