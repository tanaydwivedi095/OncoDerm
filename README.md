# OncoDerm: Streamlit-Based Phase

This phase of the OncoDerm project focuses on deploying a **Streamlit-based web application** for real-time skin cancer classification. The application integrates multiple deep learning models to provide accurate diagnostic predictions.

## Overview
- Develops an **interactive web interface** using **Streamlit**.
- Allows users to **upload images** and select different classification models.
- Utilizes **pretrained CNN models** combined with **ML classifiers** for diagnosis.
- Displays results in a **tabular format** and provides a **bar chart analysis**.

## Features
- **Multi-Model Classification**: Users can choose from **VGG16, VGG19, Xception, EfficientNetB3, EfficientNetV2B3, ResNet50, and DenseNet121**.
- **User-Friendly Interface**: Drag-and-drop image upload functionality.
- **Visual Analysis**: Results displayed with a table and bar chart for better interpretation.
- **Real-Time Processing**: Efficient computation for quick diagnosis.
- **Hybrid Model Integration**: Users can combine predictions from multiple models for improved accuracy.

## How It Works
1. User uploads a dermoscopic image.
2. Selects one or multiple models for classification.
3. The selected models preprocess the image and generate predictions.
4. Results are displayed in a structured format with a **classification summary**.
5. A bar chart visualizes the consensus among different models.

## Project Phases
1. **Transfer Learning (CNNs)**: Feature extraction using CNNs, combined with an **RFC classifier**. [🔗 Link](https://github.com/tanaydwivedi095/OncoDerm_Transfer_Learning)
2. **Transformer-Based Models**: Implementing **ViTs** for enhanced classification. [🔗 Link](https://github.com/tanaydwivedi095/OncoDerm_Transformers)
3. **Streamlit UI (Current Phase)**: Deploying an interactive application for real-time classification.

### Additional Resources
- **Hybrid Model Source Code**: [🔗 Link](https://github.com/tanaydwivedi095/Skin-Cancer-Classification)
- **Vision Transformer Model Source Code**: [🔗 Link](https://github.com/tanaydwivedi095/Skin-Cancer-Classification-using-ViT)
- **Model Files (h5 & pkl) Download**: [🔗 Link](https://drive.google.com/drive/folders/1ZNjcluKK7dDyQ7NhVVcKOy7VKnYq0z7d?usp=sharing)

## Installation & Usage
### Step 1: Clone the Repository
```bash
git clone https://github.com/tanaydwivedi095/OncoDerm_Streamlit.git
cd OncoDerm_Streamlit
```

### Step 2: Install Dependencies Manually
Since `requirements.txt` is not available, install the required packages manually:
```bash
pip install tensorflow opencv-python streamlit pillow joblib numpy pandas matplotlib scikit-learn
```

### Step 3: Download Model Weights
Before running the Streamlit app, download the necessary model weights from the provided Google Drive link:
[🔗 Model Files (h5 & pkl)](https://drive.google.com/drive/folders/1ZNjcluKK7dDyQ7NhVVcKOy7VKnYq0z7d?usp=sharing)

### Step 4: Run the Web Application
Launch the Streamlit application using:
```bash
streamlit run WEB.py
```

## Contributions
Contributions are welcome! Feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

