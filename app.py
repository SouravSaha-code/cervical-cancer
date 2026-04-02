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
from tensorflow.keras.models import load_model       # type: ignore
from tensorflow.keras.preprocessing import image     # type: ignore
from werkzeug.utils import secure_filename
import base64
import h5py

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
MAX_FILE_SIZE      = 16 * 1024 * 1024   # 16 MB
IMG_SIZE           = (224, 224)

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Model file paths ──────────────────────────────────────────────────────────
RESNET_MODEL_PATH    = 'resnet152_cervical_cancer.keras'
VGG_MODEL_PATH       = 'vgg16_cervical_cancer.keras'
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


# ── Weight loading helper ─────────────────────────────────────────────────────
def load_weights_by_shape(model, h5_path):
    """
    Loads weights from an .h5 file into a Keras model by matching shapes.
    Fixes the ordering mismatch that occurs with some checkpoint formats.
    """
    all_weight_arrays = []

    def collect_weights(grp):
        if isinstance(grp, h5py.Dataset):
            all_weight_arrays.append(np.array(grp))
        else:
            for key in grp.keys():
                if key != 'vars':
                    collect_weights(grp[key])

    with h5py.File(h5_path, 'r') as f:
        for top_key in f.keys():
            if '_layer_checkpoint_dependencies' in top_key:
                collect_weights(f[top_key])

    # Build a shape → [arrays] lookup from the file
    file_by_shape = {}
    for w in all_weight_arrays:
        shape = tuple(w.shape)
        file_by_shape.setdefault(shape, []).append(w)

    # Match file weights to model weights in model order, by shape
    matched = []
    counters = {}
    for mw in model.weights:
        shape = tuple(mw.shape)
        idx = counters.get(shape, 0)
        candidates = file_by_shape.get(shape, [])
        if idx < len(candidates):
            matched.append(candidates[idx])
        else:
            print(f"  ⚠️  No match for shape {shape} (index {idx}), keeping original")
            matched.append(mw.numpy())
        counters[shape] = idx + 1

    model.set_weights(matched)
    print(f"  ✓ Weights loaded via shape-matching from: {h5_path}")


# ── Ensemble model ────────────────────────────────────────────────────────────
class WeightedEnsembleModel:
    """Loads multiple Keras models and combines their predictions."""

    def __init__(self, model_paths, weights):
        self.models  = []
        self.weights = weights
        print("Loading models…")
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            # Try standard load first; fall back to shape-matched loading
            try:
                mdl = load_model(path, compile=False)
                print(f"  ✓ Loaded: {path}")
            except Exception as e:
                print(f"  ⚠️  Standard load failed for {path}: {e}")
                print(f"  🔄 Trying shape-matched weight loading…")

                # Step 1: Load architecture only (skip_mismatch ignores weight errors)
                try:
                    mdl = load_model(path, compile=False, skip_mismatch=True)
                except Exception:
                    # If even that fails, load with custom_objects workaround
                    import zipfile, json, tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        with zipfile.ZipFile(path, 'r') as z:
                            z.extractall(tmpdir)
                        config_path = os.path.join(tmpdir, 'config.json')
                        with open(config_path, 'r') as cf:
                            model_config = json.load(cf)
                        mdl = tf.keras.models.model_from_json(json.dumps(model_config))

                # Step 2: Load weights from the .keras file (it's a zip containing weights.h5)
                import zipfile, tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(path, 'r') as z:
                        z.extractall(tmpdir)
                    weights_h5 = os.path.join(tmpdir, 'model.weights.h5')
                    if os.path.exists(weights_h5):
                        load_weights_by_shape(mdl, weights_h5)
                    else:
                        # Try any .h5 file inside the zip
                        h5_files = [f for f in os.listdir(tmpdir) if f.endswith('.h5')]
                        if h5_files:
                            load_weights_by_shape(mdl, os.path.join(tmpdir, h5_files[0]))
                        else:
                            raise FileNotFoundError(f"No weights file found inside {path}")

                print(f"  ✓ Loaded (shape-matched): {path}")

            self.models.append(mdl)

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
    """Load ensemble_config.pkl and instantiate the weighted ensemble."""
    global ensemble_model, config

    if not os.path.exists(ENSEMBLE_CONFIG_PATH):
        print(f"❌ Config file not found: {ENSEMBLE_CONFIG_PATH}")
        return False

    try:
        with open(ENSEMBLE_CONFIG_PATH, 'rb') as f:
            config = pickle.load(f)

        # Validate required keys
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
    """Return the correct MIME type from file extension (fixes jpeg/png/tiff confusion)."""
    ext = filename.rsplit('.', 1)[-1].lower()
    return {
        'jpg':  'image/jpeg',
        'jpeg': 'image/jpeg',
        'png':  'image/png',
        'bmp':  'image/bmp',
        'tif':  'image/tiff',
        'tiff': 'image/tiff',
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
        return jsonify({'error': 'Model not loaded. Please restart the server.'}), 500

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

        # Encode with correct MIME type (was always hardcoded to jpeg — now fixed)
        mime = get_mime_type(filename)
        with open(filepath, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')

        result['image_data'] = f"data:{mime};base64,{img_b64}"
        result['filename']   = filename

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

    finally:
        # Always clean up — even if an exception was raised (was missing before)
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


@app.route('/health')
def health():
    return jsonify({
        'status':             'healthy' if ensemble_model is not None else 'unhealthy',
        'models_loaded':      ensemble_model is not None,
        'tensorflow_version': tf.__version__,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 70)
    print("  CERVICAL CANCER CELL CLASSIFICATION — FLASK APPLICATION")
    print("=" * 70)

    if load_ensemble():
        print("\n🚀 Starting Flask server…")
        print("📍 http://localhost:5000")
        print("=" * 70)
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("\n❌ Failed to load models. Ensure these files exist:")
        print(f"   • {ENSEMBLE_CONFIG_PATH}")
        print(f"   • {RESNET_MODEL_PATH}  (path stored inside the config)")
        print(f"   • {VGG_MODEL_PATH}     (path stored inside the config)")
        print("=" * 70)
