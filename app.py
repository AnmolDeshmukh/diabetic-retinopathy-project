from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import urllib.request
import gdown


# Function to download models if they don't exist
def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading model to {output_path}...")
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output=output_path, quiet=False)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

# Google Drive file IDs
final_model_id = '13s6TExlZc5TRIIYg3k_v6zEZGLtz_lhh'
macularedema_id = '1CaMPYnSaiyxrTTVsYCeVrex6IpAYm_6j'

# Download the models
download_model(final_model_id, 'final_model.pth')
download_model(macularedema_id, 'macularedema.pth')




app = Flask(__name__)

# Configuration for file upload folders
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MASKS_FOLDER'] = 'static/masks'

# Ensure these directories exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['MASKS_FOLDER']):
    os.makedirs(app.config['MASKS_FOLDER'])

# Load Model 1 for Diabetic Retinopathy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    """
    Loads and preprocesses the image from the given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input.
    """
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image.to(device)

def predict_image(model, image_path, class_names):
    """
    Predicts the class of the image using the given model.

    Args:
        model (torch.nn.Module): The trained model for prediction.
        image_path (str): Path to the image file.
        class_names (list): List of class names corresponding to model outputs.

    Returns:
        str: Predicted class name.
    """
    model.eval()
    image = load_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
    return predicted_class

# Initialize the Diabetic Retinopathy model
model_dr = models.resnet50(pretrained=False)
num_classes_dr = 5
model_dr.fc = torch.nn.Linear(model_dr.fc.in_features, num_classes_dr)
model_dr.load_state_dict(torch.load('final_model.pth', map_location=device))
model_dr.to(device)
class_names_dr = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Define a MultiTaskModel class for Macular Edema prediction
class MultiTaskModel(nn.Module):
    """
    A multi-task model based on ResNet50 for predicting Diabetic Retinopathy and Macular Edema.
    """
    def __init__(self, num_classes1, num_classes2):
        """
        Initializes the MultiTaskModel.

        Args:
            num_classes1 (int): Number of classes for the first task (Diabetic Retinopathy).
            num_classes2 (int): Number of classes for the second task (Macular Edema).
        """
        super(MultiTaskModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(self.resnet.fc.in_features, num_classes1)
        self.fc2 = nn.Linear(self.resnet.fc.in_features, num_classes2)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Outputs for both tasks.
        """
        x = self.resnet(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2

# Initialize the Macular Edema model
num_classes_me = 3
model_me = MultiTaskModel(num_classes_dr, num_classes_me)
model_me.load_state_dict(torch.load('macularedema.pth', map_location=device))
model_me.to(device)
class_names_me = ['Low Risk', 'Medium Risk', 'High Risk']

def predict_macular_edema(model, image_path):
    """
    Predicts the class of Macular Edema from the image using the given model.

    Args:
        model (torch.nn.Module): The trained multi-task model.
        image_path (str): Path to the image file.

    Returns:
        str: Predicted Macular Edema class name.
    """
    model.eval()
    image = load_image(image_path)
    with torch.no_grad():
        output_dr, output_me = model(image)
        _, predicted_me = torch.max(output_me, 1)
        predicted_class_me = class_names_me[predicted_me.item()]
    return predicted_class_me

# Define Flask routes

@app.route('/')
def welcome():
    """
    Renders the welcome page.
    """
    return render_template('welcome.html')

@app.route('/login', methods=['GET'])
def login_form():
    """
    Renders the login form.
    """
    return render_template('index.html')  # Assuming index.html is your login form

@app.route('/profile')
def profile():
    """
    Renders the user profile page.
    """
    return render_template('profile.html')

@app.route('/login', methods=['POST'])
def login():
    """
    Handles the login logic.

    If the username and password are correct, redirects to the profile page.
    Otherwise, redirects back to the login form.
    """
    username = request.form['username']
    password = request.form['password']
    if username == 'admin' and password == 'pass':
        return redirect(url_for('profile'))
    return redirect(url_for('login_form'))  # Redirect back to login form on failure

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Handles the image upload process.

    If a file is uploaded, it saves the file, performs predictions, and redirects to the loading page.
    """
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction_dr = predict_image(model_dr, filepath, class_names_dr)
            prediction_me = predict_macular_edema(model_me, filepath)
            # Redirect to the loading page with the necessary data
            return redirect(url_for('loading', filename=filename, prediction_dr=prediction_dr, prediction_me=prediction_me))
    return render_template('upload.html')

@app.route('/loading')
def loading():
    """
    Displays the loading screen with predictions.

    Renders the loading.html page and passes the filename and predictions as parameters.
    """
    filename = request.args.get('filename')
    prediction_dr = request.args.get('prediction_dr')
    prediction_me = request.args.get('prediction_me')
    return render_template('loading.html', filename=filename, prediction_dr=prediction_dr, prediction_me=prediction_me)

@app.route('/analysis')
def analysis():
    """
    Displays the analysis results.

    Renders the analysis.html page and passes the filename, predictions, and mask image if available.
    """
    filename = request.args.get('filename')
    prediction_dr = request.args.get('prediction_dr')
    prediction_me = request.args.get('prediction_me')
    mask_filename = f"{os.path.splitext(filename)[0]}_mask.jpg"
    mask_filepath = os.path.join(app.config['MASKS_FOLDER'], mask_filename)
    mask_exists = os.path.exists(mask_filepath)
    return render_template('analysis.html', filename=filename, prediction_dr=prediction_dr, prediction_me=prediction_me, mask_exists=mask_exists, mask_filename=mask_filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
