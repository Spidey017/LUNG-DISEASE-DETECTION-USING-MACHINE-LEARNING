from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
import numpy as np

# More robust TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.efficientnet import preprocess_input
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    exit(1)

from PIL import Image
import datetime
import psycopg2
from psycopg2 import pool
import functools
import atexit

app = Flask(__name__, 
            static_folder='static',  # Explicitly set static folder
            static_url_path='/static')

# Set a secret key for session management
app.secret_key = 'your_very_secure_secret_key_here'  # Change this to a random string in production

# Temporary folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define class labels
class_names = ['COVID-19','NORMAL', 'PNEUMONIA']

# Hard-coded admin credentials
ADMIN_USERNAME = "vsm"
ADMIN_PASSWORD = "aiml"

# PostgreSQL Database Configuration
# Replace these with your actual database credentials
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "lungpredict"
DB_USER = "postgres"
DB_PASSWORD = "arman786"

# Create connection pool
try:
    connection_pool = pool.SimpleConnectionPool(
        1, 10,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    print("Database connection pool created successfully")
except Exception as e:
    print(f"Error creating database connection pool: {e}")
    connection_pool = None

# Login required decorator
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view

# Load the model at startup with error handling
PRIMARY_MODEL_PATH = "respiratory_disease_classifier.keras"
FALLBACK_MODEL_PATH = "respiratory_disease_classifier.h5"
model = None

def try_load_model(path):
    try:
        if os.path.exists(path):
            print(f"Attempting to load model from: {path}")
            # Avoid compilation; safer across TF/Keras versions
            loaded = load_model(path, compile=False)
            # Log resolved input shape if available without forcing a build
            resolved_shape = getattr(loaded, 'input_shape', None)
            if not resolved_shape and getattr(loaded, 'inputs', None):
                try:
                    resolved_shape = tuple(loaded.inputs[0].shape)
                except Exception:
                    resolved_shape = None
            if resolved_shape:
                print(f"Loaded model input shape: {resolved_shape}")
                if len(resolved_shape) >= 4:
                    channel_dim = resolved_shape[-1]
                    if channel_dim == 1:
                        print("Model expects grayscale images")
                    elif channel_dim == 3:
                        print("Model expects RGB images")
            print("Model loaded successfully")
            return loaded
        else:
            print(f"Model file not found: {path}")
            return None
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")
        return None

# Try primary, then fallback
model = try_load_model(PRIMARY_MODEL_PATH)
if model is None:
    model = try_load_model(FALLBACK_MODEL_PATH)

def _get_model_input_spec():
    """Infer expected (height, width, channels) from the loaded model if available."""
    try:
        if model is None:
            return (224, 224, 3)
        shape = getattr(model, 'input_shape', None)
        if not shape and getattr(model, 'inputs', None):
            try:
                shape = tuple(model.inputs[0].shape)
            except Exception:
                shape = None
        # Typical shape: (None, H, W, C)
        if isinstance(shape, tuple) and len(shape) >= 4:
            height = int(shape[1]) if shape[1] is not None else 224
            width = int(shape[2]) if shape[2] is not None else 224
            channels = int(shape[3]) if shape[3] is not None else 3
            return (height, width, channels)
    except Exception as e:
        print(f"Could not derive model input spec: {e}")
    return (224, 224, 3)


def preprocess_image(image_path, img_size=None):
    """Load and preprocess the image for model prediction, matching the model's expected input."""
    try:
        # Determine target size and channels
        height, width, channels = _get_model_input_spec()
        if img_size is None:
            img_size = (width, height)  # PIL expects (W, H)

        # Load the image
        img = Image.open(image_path)
        print(f"Original image mode: {img.mode}, size: {img.size}")

        # Convert mode based on expected channels
        if channels == 1:
            if img.mode != 'L':
                img = img.convert('L')
            print("Converted to Grayscale (L)")
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            print("Converted to RGB")

        # Resize
        img = img.resize(img_size)
        img_array = np.array(img)

        # Ensure channel dimension
        if channels == 1 and img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)

        print(f"Image array shape before preprocessing: {img_array.shape}")

        # Apply preprocessing
        if channels == 3:
            # Use EfficientNet preprocessing which scales to [-1, 1]
            img_array = preprocess_input(img_array.astype('float32'))
        else:
            # For grayscale, simple [0,1] scaling
            img_array = img_array.astype('float32') / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        print(f"Final preprocessed shape: {img_array.shape}")
        return img_array

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None    


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to save prediction to database
def save_prediction(name, age, gender, prediction, disease_name, confidence, image_path):
    if connection_pool is None:
        print("Database connection pool not available")
        return False
    
    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO predictions 
                (name, age, gender, prediction_result, disease_name, confidence, image_path) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (name, age, gender, prediction, disease_name, confidence, image_path))
            conn.commit()
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        conn.rollback()
        return False
    finally:
        connection_pool.putconn(conn)

# Function to get all predictions from database
def get_predictions():
    if connection_pool is None:
        print("Database connection pool not available")
        return []
    
    conn = connection_pool.getconn()
    predictions = []
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                SELECT name, age, gender, prediction_result, disease_name, 
                       confidence, image_path, prediction_date 
                FROM predictions 
                ORDER BY prediction_date DESC
            ''')
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                prediction = dict(zip(columns, row))
                # Convert the datetime object to string for template rendering
                prediction['prediction_date'] = prediction['prediction_date'].strftime("%Y-%m-%d %H:%M")
                predictions.append(prediction)
    except Exception as e:
        print(f"Error retrieving predictions: {e}")
    finally:
        connection_pool.putconn(conn)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/user_details')
def user_details():
    return render_template('user_details.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Get user details from the form
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    
    # Pass these details to the upload image page
    return render_template('upload_image.html', name=name, age=age, gender=gender)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check credentials against hardcoded values
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('history'))
        else:
            error = 'Invalid username or password. Please try again.'
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    # Get prediction history from database
    history_data = get_predictions()
    return render_template('history.html', history_data=history_data, username=session.get('username'))

@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if model is loaded
    if model is None:
        return "Model not loaded. Cannot make predictions. Please check server logs.", 500
    
    # Collect user details from hidden fields
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')

    # Check if an image is uploaded
    if 'image' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Real prediction logic using the deep learning model
        try:
            # Preprocess the uploaded image
            processed_image = preprocess_image(filepath)
            
            if processed_image is None:
                return "Error processing the image", 500
            
            # Make prediction
            prediction = model.predict(processed_image)
            raw_output = prediction[0]
            print(f"Raw model output: {raw_output.tolist()}")

            # Ensure probabilities via softmax if outputs are logits
            try:
                probs = tf.nn.softmax(raw_output).numpy()
            except Exception:
                probs = raw_output
            print(f"Probabilities: {probs.tolist()}")

            # Guard against class mapping mismatch
            if len(class_names) != len(probs):
                print(f"WARNING: class_names length ({len(class_names)}) does not match model outputs ({len(probs)}).")
                auto_labels = [f"Class {i}" for i in range(len(probs))]
                used_class_names = auto_labels
            else:
                used_class_names = class_names

            predicted_class_index = int(np.argmax(probs))
            disease_name = used_class_names[predicted_class_index]
            confidence = float(probs[predicted_class_index]) * 100
            
            # Determine positive/negative result
            prediction_result = "Positive" if disease_name != "NORMAL" else "Negative"
            
            # Get relative image path for database storage
            relative_image_path = f'/static/uploads/{filename}'
            
            # Save the prediction to database
            confidence_str = f"{confidence:.2f}%"
            save_prediction(name, int(age), gender, prediction_result, disease_name, 
                           confidence_str, relative_image_path)
            
            return render_template('prediction.html', 
                                name=name, 
                                age=age, 
                                gender=gender, 
                                image_url=relative_image_path, 
                                prediction=prediction_result, 
                                disease_name=disease_name,
                                confidence=confidence_str)
        except Exception as e:
            print(f"Error in prediction: {e}")
            return f"Error in prediction: {str(e)}", 500

    return "Invalid file format", 400


def close_db_pool_on_exit():
    if connection_pool:
        try:
            connection_pool.closeall()
        except Exception as e:
            print(f"Error closing database pool: {e}")

atexit.register(close_db_pool_on_exit)

# Add a health check route
@app.route('/health')
def health_check():
    status = {
        'tensorflow': tf.__version__ if 'tf' in globals() else 'Not loaded',
        'model': 'Loaded' if model is not None else 'Not loaded',
        'database': 'Connected' if connection_pool is not None else 'Not connected'
    }
    return status

if __name__ == '__main__':
    print("="*50)
    print("LUNG DISEASE DETECTION APP")
    print("="*50)
    print(f"TensorFlow: {'✓ Loaded' if 'tf' in globals() else '✗ Not loaded'}")
    print(f"Model: {'✓ Loaded' if model is not None else '✗ Not loaded'}")
    print(f"Database: {'✓ Connected' if connection_pool is not None else '✗ Not connected'}")
    print("="*50)
    
    # Only run the app if critical components are loaded
    if model is None:
        print("WARNING: Model not loaded. The app will run but predictions won't work.")
        print("Please check that 'respiratory_disease_classifier.keras' exists in the current directory.")
    
    app.run(debug=True)