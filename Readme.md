
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
- [Contributing](#contributing)
- [License](#license)

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
   git clone https://github.com/your-repo/retinal-disease-detection.git
   cd retinal-disease-detection
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

- **Macular Edema Model**:
  - Custom multi-task model based on ResNet50 to predict both Diabetic Retinopathy and Macular Edema.
  - Loaded from `macularedema.pth`.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests with your improvements or bug fixes.
