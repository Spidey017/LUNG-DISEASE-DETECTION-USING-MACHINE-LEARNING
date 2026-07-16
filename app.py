
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
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

from PIL import Image, ImageOps
import datetime
import psycopg2
from psycopg2 import pool
import functools
import atexit
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from io import BytesIO

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

# Define precautions for each disease
precautions_map = {
    'NORMAL': {
        'title': '🩺 NORMAL LUNG (Healthy Lungs)',
        'goal': 'Keep lungs clean, strong, and infection-free.',
        'precautions': [
            {'icon': '🚭', 'title': 'Avoid Smoking', 'description': 'main cause of chronic lung damage and infections.'},
            {'icon': '🏃‍♂️', 'title': 'Exercise Regularly', 'description': 'improves lung strength and oxygen flow.'},
            {'icon': '🌫️', 'title': 'Avoid Air Pollution', 'description': 'stay away from dust, smoke, and toxic fumes.'},
            {'icon': '🥦', 'title': 'Eat a Balanced Diet', 'description': 'rich in antioxidants (Vitamin C, E, zinc).'},
            {'icon': '💉', 'title': 'Get Vaccinated', 'description': 'Flu + Pneumococcal vaccines - prevents common respiratory infections.'}
        ]
    },
    'PNEUMONIA': {
        'title': '🫁 PNEUMONIA (Non-COVID)',
        'goal': 'Prevent respiratory infections and support strong immune defense.',
        'precautions': [
            {'icon': '💉', 'title': 'Take Vaccines', 'description': 'Pneumonia & Flu Vaccines - main preventive defense.'},
            {'icon': '🧼', 'title': 'Maintain Good Hygiene', 'description': 'wash hands often, avoid touching face.'},
            {'icon': '🤧', 'title': 'Avoid Close Contact', 'description': 'stay away from sick people with cough or fever.'},
            {'icon': '🚭', 'title': 'Quit Smoking & Limit Alcohol', 'description': 'strengthens lung immunity.'},
            {'icon': '💊', 'title': 'Treat Colds or Flu Early', 'description': 'prevents infection from reaching the lungs.'}
        ]
    },
    'COVID-19': {
        'title': '🦠 COVID-19',
        'goal': 'Prevent viral transmission and protect community health.',
        'precautions': [
            {'icon': '💉', 'title': 'Get Fully Vaccinated', 'description': 'and Boosters - reduces risk of severe illness.'},
            {'icon': '😷', 'title': 'Wear a Mask in Public', 'description': 'especially in crowded or indoor places.'},
            {'icon': '🧴', 'title': 'Wash/Sanitize Hands Frequently', 'description': 'stop virus spread via surfaces.'},
            {'icon': '↔️', 'title': 'Maintain Social Distance', 'description': '1–2 meters - prevent airborne transmission.'},
            {'icon': '🏠', 'title': 'Isolate if Symptomatic', 'description': 'protect others and control spread.'}
        ]
    }
}


def get_prediction_reason(predicted_index, probs, class_labels=None):
    """Return a structured explanation for the predicted class.

    Returns a dict with keys:
      - text: a short non-clinical explanation string (includes basic statistics)
      - features: a dict mapping feature names (Lung Fields, Distribution, ...) to
                  the expected observation for the predicted class (from a
                  knowledge table). These are for display only and not a
                  clinical diagnosis.
    """
    try:
        probs = np.asarray(probs, dtype=float)
    except Exception:
        return {'text': 'No explanation available.', 'features': {}}

    # Defensive labels
    if class_labels is None or len(class_labels) != len(probs):
        class_labels = [f"Class {i}" for i in range(len(probs))]

    # Short, cautious domain-agnostic descriptions
    desc_map = {
        'COVID-19': 'The model found image features that were associated with COVID-19 cases in the training data.',
        'PNEUMONIA': 'The model detected patterns commonly seen in pneumonia cases (based on training examples).',
        'NORMAL': 'The model did not find strong abnormalities compared with images labeled as normal in the training data.'
    }

    # Feature-level guidance table (based on the table you provided)
    feature_table = {
        'COVID-19': {
            'Lung Fields': 'Hazy, cloudy',
            'Distribution': 'Diffuse, both sides',
            'Opacity Type': 'Patchy / ground-glass',
            'Pleural Effusion': 'Rare',
            'Borders': 'Slightly blurred overall',
            'Symmetry': 'Bilateral symmetry',
            'Texture': 'Hazy diffuse',
            'AI Focus Area': 'Both sides (peripheral)'
        },
        'PNEUMONIA': {
            'Lung Fields': 'White patches',
            'Distribution': 'Localized, one side',
            'Opacity Type': 'Dense, lobar',
            'Pleural Effusion': 'Sometimes',
            'Borders': 'Blurred near infection',
            'Symmetry': 'Asymmetrical',
            'Texture': 'Rough localized',
            'AI Focus Area': 'Central or one side'
        },
        'NORMAL': {
            'Lung Fields': 'Clear, dark',
            'Distribution': 'Even, bilateral',
            'Opacity Type': 'None',
            'Pleural Effusion': 'Absent',
            'Borders': 'Sharp',
            'Symmetry': 'Symmetrical',
            'Texture': 'Smooth',
            'AI Focus Area': 'Low activation'
        }
    }

    # Build statistical context
    sorted_idx = np.argsort(probs)[::-1]
    top = int(sorted_idx[0])
    sec = int(sorted_idx[1]) if len(sorted_idx) > 1 else None
    top_conf = float(probs[top]) * 100.0
    sec_conf = float(probs[sec]) * 100.0 if sec is not None else None

    label = class_labels[top]
    second_label = class_labels[sec] if sec is not None else None

    base_desc = desc_map.get(label, '')
    stat = f"The classifier assigned {top_conf:.2f}% probability to {label}"
    if second_label is not None:
        stat += f" compared with {sec_conf:.2f}% for {second_label}."
    else:
        stat += '.'

    explanation = f"{base_desc} {stat} This is an AI statistical prediction and not a clinical diagnosis. Please consult a healthcare professional for confirmation."

    features = feature_table.get(label, {})
    return {'text': explanation, 'features': features}

# Hard-coded admin credentials
ADMIN_USERNAME = "vsm"
ADMIN_PASSWORD = "aiml"

# PostgreSQL Database Configuration
# Replace these with your actual database credentials
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "lungdeases"
DB_USER = "postgres"
DB_PASSWORD = "kamran"

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

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

def build_efficientnet_model():
    input_tensor = Input(shape=(224, 224, 3))  # Use 224x224 for RGB
    base_model = EfficientNetB0(include_top=False, input_tensor=input_tensor, pooling='avg')
    x = Dense(3, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def try_load_model(path):
    try:
        if os.path.exists(path):
            print(f"Attempting to load model from: {path}")
            loaded = None
            try:
                loaded = load_model(path, compile=False)
                print("Model loaded successfully with standard approach")
            except Exception as e1:
                print(f"Standard load failed: {e1}")
                # Try rebuilding architecture for RGB weights (224x224)
                try:
                    input_tensor = Input(shape=(224, 224, 3))
                    base_model = EfficientNetB0(include_top=False, input_tensor=input_tensor, pooling='avg')
                    x = Dense(3, activation='softmax')(base_model.output)
                    loaded = Model(inputs=base_model.input, outputs=x)
                    loaded.load_weights(path)
                    print("Model loaded with rebuilt EfficientNetB0 architecture (RGB, 224x224)")
                except Exception as e2:
                    print(f"Custom load failed: {e2}")
                    return None
            return loaded
        else:
            print(f"Model file not found: {path}")
            return None
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")
        return None

PRIMARY_MODEL_PATH = "respiratory_disease_classifier.keras"
FALLBACK_MODEL_PATH = "respiratory_disease_classifier.h5"

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
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))

        # Convert mode based on expected channels
        if channels == 1:
            if img.mode != 'L':
                img = img.convert('L')
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')

        # Resize
        img = img.resize(img_size)
        img_array = np.array(img)

        # Ensure channel dimension
        if channels == 1 and img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)

        # Apply preprocessing
        if channels == 3:
            # Use EfficientNet preprocessing which scales to [-1, 1]
            img_array = preprocess_input(img_array.astype('float32'))
        else:
            # For grayscale, simple [0,1] scaling
            img_array = img_array.astype('float32') / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

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

def generate_pdf_report(name, age, gender, disease_name, confidence, prediction_result, 
                       explanation_text, precautions, image_path=None):
    """Generate a PDF report of the prediction"""
    try:
        # Create PDF buffer
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                              topMargin=0.5*inch, bottomMargin=0.5*inch,
                              leftMargin=0.75*inch, rightMargin=0.75*inch)
        
        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#4CAF50'),
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=8
        )
        
        # Title
        elements.append(Paragraph("🫁 Lung Disease Detection Report", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Report Date
        report_date = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
        elements.append(Paragraph(f"<b>Report Generated:</b> {report_date}", normal_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Patient Information Section
        elements.append(Paragraph("👤 Patient Information", heading_style))
        patient_data = [
            ['Full Name:', name],
            ['Age:', str(age)],
            ['Gender:', gender],
            ['Report Date:', report_date]
        ]
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F5E9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(patient_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Prediction Results Section
        elements.append(Paragraph("🔬 Prediction Results", heading_style))
        
        # Determine result color and badge
        if prediction_result == 'Positive':
            result_text = f"✓ {disease_name} DETECTED"
        else:
            result_text = f"✓ NO ABNORMALITIES DETECTED"
        
        prediction_data = [
            ['Diagnosis:', result_text],
            ['Confidence Score:', f"{confidence}"],
            ['Status:', "Requires Medical Consultation" if prediction_result == 'Positive' else "Monitor Health"]
        ]
        prediction_table = Table(prediction_data, colWidths=[2*inch, 4*inch])
        prediction_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#FFF3E0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(prediction_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Clinical Explanation
        elements.append(Paragraph("📋 Clinical Explanation", heading_style))
        elements.append(Paragraph(explanation_text, normal_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Precautions Section
        if precautions:
            elements.append(Paragraph("🛡️ Recommended Precautions", heading_style))
            elements.append(Paragraph(f"<b>Goal: {precautions.get('goal', '')}</b>", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Precautions list
            precautions_list = precautions.get('precautions', [])
            for i, precaution in enumerate(precautions_list, 1):
                icon = precaution.get('icon', '')
                title = precaution.get('title', '')
                description = precaution.get('description', '')
                precaution_text = f"<b>{i}. {icon} {title}:</b> {description}"
                elements.append(Paragraph(precaution_text, normal_style))
                elements.append(Spacer(1, 0.08*inch))
            
            elements.append(Spacer(1, 0.2*inch))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#D32F2F'),
            alignment=TA_CENTER,
            spaceAfter=12,
            fontName='Helvetica-BoldOblique'
        )
        elements.append(Paragraph(
            "⚠️ <b>DISCLAIMER:</b> This report is AI-assisted and not a substitute for professional medical advice. "
            "Always consult with a qualified healthcare professional for definitive diagnosis and treatment.",
            disclaimer_style
        ))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("LungPredict - Lung Disease Detection System", footer_style))
        elements.append(Paragraph(f"Generated on {report_date}", footer_style))
        
        # Build PDF
        doc.build(elements)
        pdf_buffer.seek(0)
        return pdf_buffer
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None

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

        try:
            # Helper: convert model raw output to probabilities only when needed
            def to_probs(raw):
                raw = np.asarray(raw, dtype=np.float32)
                s = raw.sum()
                if raw.min() >= 0.0 and raw.max() <= 1.0 and abs(s - 1.0) < 1e-3:
                    # already probabilities
                    return raw
                # assume logits -> softmax
                try:
                    return tf.nn.softmax(raw).numpy()
                except Exception:
                    # fallback numeric softmax
                    ex = np.exp(raw - np.max(raw))
                    return ex / ex.sum()

            # Helper: preprocess a PIL image and return model-ready array
            def preprocess_pil_image(pil_img):
                # Use same _get_model_input_spec as rest of app
                h, w, channels = _get_model_input_spec()
                # ensure RGB or L mode per channels
                if channels == 1:
                    if pil_img.mode != 'L':
                        pil_img = pil_img.convert('L')
                else:
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                pil_img = pil_img.resize((w, h))
                arr = np.array(pil_img).astype('float32')
                if channels == 3:
                    arr = preprocess_input(arr)
                else:
                    arr = arr / 255.0
                    if arr.ndim == 2:
                        arr = np.expand_dims(arr, axis=-1)
                return arr

            # Create simple Test-Time Augmentations (TTA) to stabilize / sharpen prediction
            pil = Image.open(filepath)
            base = pil.convert('RGB').resize((224, 224))  # consistent base size

            variants = [
                base,
                ImageOps.mirror(base),
                ImageOps.flip(base),
                base.rotate(15),
                base.rotate(-15)
            ]

            batch = np.stack([preprocess_pil_image(v) for v in variants], axis=0)

            # Predict on the TTA batch and average probabilities
            preds = model.predict(batch, verbose=0)  # shape (n_variants, n_classes)
            probs_list = []
            for out in preds:
                probs_list.append(to_probs(out))
            avg_probs = np.mean(probs_list, axis=0)

            print(f"TTA probabilities (avg): {avg_probs.tolist()}")

            # Guard against class mapping mismatch
            if len(class_names) != len(avg_probs):
                print(f"WARNING: class_names length ({len(class_names)}) does not match model outputs ({len(avg_probs)}).")
                used_class_names = [f"Class {i}" for i in range(len(avg_probs))]
            else:
                used_class_names = class_names

            predicted_class_index = int(np.argmax(avg_probs))
            disease_name = used_class_names[predicted_class_index].strip()
            confidence = float(avg_probs[predicted_class_index]) * 100

            prediction_result = "Positive" if disease_name != "NORMAL" else "Negative"
            relative_image_path = f'/static/uploads/{filename}'
            confidence_str = f"{confidence:.2f}%"

            # Build a short explanation for the UI to show why the model made this
            # prediction (statistical + non-clinical description). This is
            # intentionally cautious and not a replacement for medical advice.
            try:
                explanation = get_prediction_reason(predicted_class_index, avg_probs, used_class_names)
                explanation_text = explanation.get('text', 'No explanation available.')
                explanation_features = explanation.get('features', {})
            except Exception as e:
                print(f"Warning: failed to build explanation: {e}")
                explanation_text = "No explanation available."
                explanation_features = {}

            # Get precautions for the predicted disease
            precautions = precautions_map.get(disease_name, {})
            print(f"DEBUG: disease_name = {disease_name}")
            print(f"DEBUG: precautions = {precautions}")

            # Save prediction (if DB available)
            try:
                save_prediction(name, int(age), gender, prediction_result, disease_name, 
                            confidence_str, relative_image_path)
            except Exception as e:
                print(f"Warning: failed to save prediction: {e}")

            return render_template('prediction.html', 
                                name=name, 
                                age=age, 
                                gender=gender, 
                                image_url=relative_image_path, 
                                prediction=prediction_result, 
                                disease_name=disease_name,
                                confidence=confidence_str,
                                explanation_text=explanation_text,
                                explanation_features=explanation_features,
                                precautions=precautions)
        except Exception as e:
            print(f"Error in prediction: {e}")
            return f"Error in prediction: {str(e)}", 500

    return "Invalid file format", 400

@app.route('/download_report', methods=['POST'])
def download_report():
    """Download prediction report as PDF"""
    try:
        # Get data from request
        data = request.get_json()
        name = data.get('name', 'Unknown')
        age = data.get('age', 'Unknown')
        gender = data.get('gender', 'Unknown')
        disease_name = data.get('disease_name', 'Unknown')
        confidence = data.get('confidence', 'Unknown')
        prediction_result = data.get('prediction_result', 'Unknown')
        explanation_text = data.get('explanation_text', 'No explanation available')
        precautions_data = data.get('precautions', {})
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(name, age, gender, disease_name, confidence, 
                                        prediction_result, explanation_text, precautions_data)
        
        if pdf_buffer:
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"LungPrediction_Report_{timestamp}.pdf"
            
            return send_file(
                pdf_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        else:
            return "Error generating PDF", 500
            
    except Exception as e:
        print(f"Error in download_report: {e}")
        return f"Error: {str(e)}", 500

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