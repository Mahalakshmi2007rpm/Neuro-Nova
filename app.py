# app.py
from flask import Flask, render_template, request, redirect, url_for, session
import os
import logging
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")
# Render and similar hosts reliably allow writes in the temp directory.
UPLOAD_FOLDER = os.getenv(
    "UPLOAD_FOLDER",
    os.path.join(tempfile.gettempdir(), "neuro_nova_uploads"),
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO)

# ---- Simple user database (demo) ----
USERS = {
    "lakshmi123": "12345",
    "admin": "admin"
}

# -------- Routes --------
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS and USERS[username] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid username or password"
    return render_template('login.html', error=error)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS:
            error = "User already exists"
        else:
            USERS[username] = password
            return redirect(url_for('login'))
    return render_template('signup.html', error=error)

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    try:
        if 'username' not in session:
            return redirect(url_for('login'))
        result = None
        error = None
        if request.method == 'POST':
            file = request.files.get('file')
            if file and file.filename:
                filename = secure_filename(file.filename)
                # Predict and generate Grad-CAM + segmentation
                try:
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    from ml.model import predict_image
                    pred_class, confidence, heatmap_path, seg_path = predict_image(filepath)
                    result = {
                        "filename": filename,
                        "pred_class": pred_class,
                        "confidence": confidence,
                        "heatmap": os.path.basename(heatmap_path),
                        "segmentation": os.path.basename(seg_path)
                    }
                    return render_template('result.html', result=result)
                except Exception as exc:
                    app.logger.exception("Upload inference failed")
                    error = str(exc)
            elif request.method == 'POST':
                error = "Please select an image file before uploading."
        return render_template('upload.html', error=error)
    except Exception:
        app.logger.exception("Upload route failed")
        return render_template(
            'upload.html',
            error='Server error while processing your request. Please try again.',
        ), 200


@app.errorhandler(500)
def internal_server_error(_error):
    return render_template(
        'upload.html',
        error='Server error occurred. Please try again in a moment.',
    ), 200
    
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# -------- Run App --------
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )