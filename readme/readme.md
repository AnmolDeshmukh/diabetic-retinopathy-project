
# Retinal Disease Detection Web Application

## Overview

This project is a Flask-based web application designed to assist ophthalmologists and healthcare professionals in detecting and analyzing retinal diseases, specifically Diabetic Retinopathy and Macular Edema. The application allows users to upload retina images, perform disease severity predictions using deep learning models, and view the results on a user-friendly interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Model Details](#model-details)
- [Image Processing and Analysis](#image-processing-and-analysis)
- [Dataset Details](#dataset-details)
- [Contributing](#contributing)


## Features

- **User Authentication**: Secure login and sign-up functionality.
- **Image Upload**: Users can upload retina images for analysis.
- **Real-Time Analysis**: Predicts the severity of Diabetic Retinopathy and Macular Edema.
- **Mask Overlay**: Displays a mask overlay on the retina image to highlight areas of concern.
- **Interactive Interface**: Toggle buttons to show/hide the mask and legend information.
- **Profile Management**: Users can view their profile, total patients, and analysis statistics.

## Project Structure

```
├── app.py                   # Main Flask application
├── static/                  # Static files (CSS, images, etc.)
│   ├── css/
│   │   ├── analysis.css
│   │   ├── welcome.css
│   │   ├── upload.css
│   │   ├── profile.css
│   │   ├── loading.css
│   │   └── styles.css
│   ├── images/
│   ├── logo.png
│   └── bg.jpeg
├── templates/               # HTML templates
│   ├── index.html
│   ├── analysis.html
│   ├── welcome.html
│   ├── upload.html
│   ├── profile.html
│   └── loading.html
├── models/                  # Pre-trained model files
│   ├── final_model.pth
│   └── macularedema.pth
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

## Setup and Installation

1. **Clone the repository**:
   ```
   git clone https://gitlab.com/amds-conference/dr-project.git
   cd dr-project
   ```

2. **Create a virtual environment**:
   ```
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Set up directories**:
   Ensure that the following directories exist in the root folder:
   - `static/uploads/`
   - `static/masks/`

5. **Run the application**:
   ```
   flask run
   ```
   The application will be available at `http://127.0.0.1:5000/`.

## Usage

1. **Login/Sign Up**: 
   - Access the login page at `http://127.0.0.1:5000/`.
   - New users can sign up, while existing users can log in with their credentials.

2. **Upload Image**: 
   - After logging in, navigate to the upload page and select a retina image to analyze.

3. **View Analysis**: 
   - Once the image is processed, the application displays the severity predictions for Diabetic Retinopathy and Macular Edema.
   - The mask overlay and legend provide a visual explanation of the detected areas of concern.

4. **Profile and History**: 
   - Users can view their profile, statistics, and analysis history.

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: PyTorch, ResNet50
- **Other Tools**: FileReader API for image preview, Flask-Login for authentication

## Model Details

- **Diabetic Retinopathy Model**:
  - Pre-trained ResNet50 model fine-tuned for classifying Diabetic Retinopathy.
  - Loaded from `final_model.pth`.
  - **Architecture**: The original fully connected layer of ResNet50, which was designed for 1000 classes in ImageNet, is replaced with a new fully connected layer that outputs predictions for five classes of diabetic retinopathy:
    - Class 0: No Diabetic Retinopathy (No DR)
    - Class 1: Mild Diabetic Retinopathy
    - Class 2: Moderate Diabetic Retinopathy
    - Class 3: Severe Diabetic Retinopathy
    - Class 4: Proliferative Diabetic Retinopathy
  - **Training**: The model was fine-tuned on a dataset specific to retinal images to ensure it could accurately classify the severity of diabetic retinopathy.
  - **Inference**: The model processes an input retinal image, passing it through several convolutional layers to extract features. These features are then flattened and passed through the final fully connected layer to generate a prediction, indicating the severity of diabetic retinopathy.

- **Macular Edema Model**:
  - Custom multi-task model based on ResNet50 to predict both Diabetic Retinopathy and Macular Edema.
  - Loaded from `macularedema.pth`.
  - **Architecture**: The base model is again ResNet50, but with two separate fully connected layers:
    - fc1: Predicts the severity of diabetic retinopathy.
    - fc2: Predicts the severity of macular edema with three classes: 
         1. Low Risk
         2. Medium Risk
         3. High Risk
  - **Multi-Task Learning**: By using a shared ResNet backbone, the model can leverage common features between the two tasks, improving overall performance and efficiency.
  - **Inference**: During inference, the image is passed through the shared ResNet layers. The output from the final convolutional layer is then passed through two fully connected layers, each producing predictions for diabetic retinopathy and macular edema, respectively.

## Image Processing and Analysis

- **Masks Folder:**
  - The `masks` folder contains retinal images with their optic disc, hard exudates, and soft exudates masks overlayed on them. These masks help highlight specific regions of the retina that are of clinical interest.
  - The folder `images_with_masks_available` contains the original retinal images without any mask overlays. These images are the raw input data used for prediction before any mask or segmentation is applied.

- **Image Preprocessing:**
  - Images are resized to 224x224 pixels, normalized using the standard ImageNet mean and standard deviation, and then converted into tensors before being fed into the models.

## Dataset Details

- **Diabetic Retinopathy Dataset** (`trainLabels.csv`):
  - This is a sample of the main dataset.
  - This dataset contains labels for retinal images indicating the level of diabetic retinopathy. It was used to train the diabetic retinopathy model.
  - The `image` column lists the image filenames, and the `level` column indicates the severity of diabetic retinopathy on a scale from 0 (No DR) to 4 (Proliferative DR).

- **Macular Edema Dataset** (`IDRiD_Disease_Grading_Training_Labels.csv`):
   - This is a sample of the main dataset.   
  - This dataset contains labels for both diabetic retinopathy and the risk of macular edema. It was used to train the macular edema model.
  - The `Image name` column lists the image filenames, `Retinopathy grade` indicates the severity of diabetic retinopathy, and `Risk of macular edema` indicates the risk level of macular edema on a scale from 0 (Low Risk) to 2 (High Risk).

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests with your improvements or bug fixes.
```