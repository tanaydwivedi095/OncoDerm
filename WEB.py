import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from PIL import Image
import cv2
import joblib
import pandas as pd


def main():
    st.title("Skin Cancer Classification using Hybrid Model")
    st.write('Upload any image that you want to classify')
    file = st.file_uploader('PLease Upload the Image', type=['jpg', '.jpeg', '.png'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width = False)
        option = st.selectbox("Please select the model to be used for prediction!", ("VGG16", "VGG19", "EfficientNetB3", "EfficientNetV2B3", "Xception", "DenseNet121","ResNet50", "All models"))
        st.write(f'You selected {option}')
        if st.button("Click here to proceed"):
            if option=="VGG16":
                temp = VGG16_logic(file)
                if temp == 0:
                    st.warning("The image has been classified as BENIGN", icon="⚠️")
                else:
                    st.warning("The image has been classified as MALIGNANT", icon="⚠️")
            elif option=="VGG19":
                temp = VGG19_logic(file)
                if temp == 0:
                    st.warning("The image has been classified as BENIGN", icon="⚠️")
                else:
                    st.warning("The image has been classified as MALIGNANT", icon="⚠️")
            elif option=="EfficientNetB3":
                temp = EfficientNetB3_logic(file)
                if temp == 0:
                    st.warning("The image has been classified as BENIGN", icon="⚠️")
                else:
                    st.warning("The image has been classified as MALIGNANT", icon="⚠️")
            elif option=="EfficientNetV2B3":
                temp = EfficientNetV2B3_logic(file)
                if temp == 0:
                    st.warning("The image has been classified as BENIGN", icon="⚠️")
                else:
                    st.warning("The image has been classified as MALIGNANT", icon="⚠️")
            elif option=="DenseNet121":
                temp = DenseNet121_logic(file)
                if temp == 0:
                    st.warning("The image has been classified as BENIGN", icon="⚠️")
                else:
                    st.warning("The image has been classified as MALIGNANT", icon="⚠️")
            elif option=="ResNet50":
                temp = ResNet50_logic(file)
                if temp == 0:
                    st.warning("The image has been classified as BENIGN", icon="⚠️")
                else:
                    st.warning("The image has been classified as MALIGNANT", icon="⚠️")
            elif option=="Xception":
                temp = Xception_logic(file)
                if temp == 0:
                    st.warning("The image has been classified as BENIGN", icon="⚠️")
                else:
                    st.warning("The image has been classified as MALIGNANT", icon="⚠️")
            else:
                All_logic(file)
    else:
        st.warning("You have not uploaded the image yet", icon="⚠️")

def All_logic(file):
    arr = [VGG16_logic(file), VGG19_logic(file), Xception_logic(file), EfficientNetB3_logic(file), EfficientNetV2B3_logic(file), ResNet50_logic(file), DenseNet121_logic(file)]
    dataDict = dict({
        "VGG16" : "Malignant⚠️" if arr[0]==1 else "Benign⚠️",
        "VGG19" : "Malignant⚠️" if arr[1]==1 else "Benign⚠️",
        "Xception" : "Malignant⚠️" if arr[2]==1 else "Benign⚠️",
        "EfficientNetB3" : "Malignant⚠️" if arr[3]==1 else "Benign⚠️",
        "EfficientNetV2B3" : "Malignant⚠️" if arr[4]==1 else "Benign⚠️",
        "ResNet50" : "Malignant⚠️" if arr[5]==1 else "Benign⚠️",
        "DenseNet121" : "Malignant⚠️" if arr[6]==1 else "Benign⚠️",
    })
    st.dataframe(data = dataDict)
    

def VGG16_logic(file):
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    
    CNN_model = tf.keras.models.load_model('CNN_VGG16.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('ML_VGG16.pkl')
    prediction_RF = ML_model.predict(features)
    return prediction_RF

def VGG19_logic(file):
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    
    CNN_model = tf.keras.models.load_model('CNN_VGG16.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('ML_VGG16.pkl')
    prediction_RF = ML_model.predict(features)
    return prediction_RF

def Xception_logic(file):
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    
    CNN_model = tf.keras.models.load_model('CNN_VGG16.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('ML_VGG16.pkl')
    prediction_RF = ML_model.predict(features)
    return prediction_RF

def EfficientNetB3_logic(file):
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    
    CNN_model = tf.keras.models.load_model('CNN_VGG16.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('ML_VGG16.pkl')
    prediction_RF = ML_model.predict(features)
    return prediction_RF

def EfficientNetV2B3_logic(file):
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    
    CNN_model = tf.keras.models.load_model('CNN_VGG16.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('ML_VGG16.pkl')
    prediction_RF = ML_model.predict(features)
    return prediction_RF

def DenseNet121_logic(file):
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    
    CNN_model = tf.keras.models.load_model('CNN_VGG16.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('ML_VGG16.pkl')
    prediction_RF = ML_model.predict(features)
    return prediction_RF

def ResNet50_logic(file):
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    
    CNN_model = tf.keras.models.load_model('CNN_VGG16.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('ML_VGG16.pkl')
    prediction_RF = ML_model.predict(features)
    return prediction_RF


if __name__ == "__main__":
    main()