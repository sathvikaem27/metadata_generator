from flask import Flask, request, render_template, redirect
import os
from werkzeug.utils import secure_filename
from utils import generate_metadata

# Initialize Flask app and specify templates directory
app = Flask(__name__, template_folder='templates')

# Correctly use __file__ (double underscores)
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'uploads'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        metadata = generate_metadata(file_path)
        os.remove(file_path)

        return render_template('results.html', metadata=metadata)

    return render_template('index.html')

# ✅ Correct if-statement to run app
if __name__ == '__main__':
    print("Running from:", os.getcwd())  # Optional debug print
    print("TEMPLATES FOUND:", os.listdir('templates'))  # Optional debug print
    app.run(debug=True)
