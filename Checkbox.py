from re import L
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
        with st.container():
            col1, col2, col3 = st.columns(3)
            image = Image.open(file)
            with col1:
                vgg16 = st.checkbox("VGG16 model")
                vgg19 = st.checkbox("VGG19 model")
                xception = st.checkbox("Xception model")
            with col2:
                efficientnetb3 = st.checkbox("EfficientNetB3 model")
                efficientnetv2b3 = st.checkbox("EfficientNetV2B3 model")
            with col3:
                resnet50 = st.checkbox("ResNet50 model")
                densenet121 = st.checkbox("DenseNet121 model")

        models = [vgg16, vgg19, xception, efficientnetb3, efficientnetv2b3, resnet50, densenet121]
        arr = [None for i in range(0,len(models))]

        if models[0]:
            arr[0] = VGG16_logic(file)
        if models[1]:
            arr[1] = VGG19_logic(file)
        if models[2]:
            arr[2] = Xception_logic(file)
        if models[3]:
            arr[3] = EfficientNetB3_logic(file)
        if models[4]:
            arr[4] = EfficientNetV2B3_logic(file)
        if models[5]:
            arr[5] = ResNet50_logic(file)
        if models[6]:
            arr[6] = DenseNet121_logic(file)

        with st.container():
            with st.container():
                displayer(arr)
            with st.container():
                charter(arr)
    else:
        st.warning("You have not uploaded the image yet", icon="⚠️")


def charter(arr):
    freq = {"Malignant":0 , "Benign":0}
    for i in arr:
        if i==1:
            freq["Malignant"] += 1
        if i==0:
            freq["Benign"] += 1
    # st.dataframe(data=freq)
    st.bar_chart(data=freq)


def displayer(arr):
    dataDictCol1 = dict({
        "VGG16" : "Malignant⚠️" if arr[0]==1 else "Benign⚠️" if arr[0]==0 else "N/A",
        "VGG19" : "Malignant⚠️" if arr[1]==1 else "Benign⚠️" if arr[1]==0 else "N/A",
        "Xception" : "Malignant⚠️" if arr[2]==1 else "Benign⚠️" if arr[2]==0 else "N/A"
    })
    dataDictCol2 = dict({
        "EfficientNetB3" : "Malignant⚠️" if arr[3]==1 else "Benign⚠️" if arr[3]==0 else "N/A",
        "EfficientNetV2B3" : "Malignant⚠️" if arr[4]==1 else "Benign⚠️" if arr[4]==0 else "N/A"
    })
    dataDictCol3 = dict({
        "ResNet50" : "Malignant⚠️" if arr[5]==1 else "Benign⚠️" if arr[5]==0 else "N/A",
        "DenseNet121" : "Malignant⚠️" if arr[6]==1 else "Benign⚠️" if arr[6]==0 else "N/A"
    })
    col1, col2, col3 = st.columns(3)
    with col1:
        st.dataframe(data = dataDictCol1)
    with col2:
        st.dataframe(data = dataDictCol2)
    with col3:
        st.dataframe(data = dataDictCol3)
    

def VGG16_logic(file):
    image = Image.open(file)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = np.expand_dims(image, axis=0)
    image = image/255.0
    
    CNN_model = tf.keras.models.load_model('VGG16/CNN_VGG16.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('VGG16/ML_VGG16.pkl')
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
    
    CNN_model = tf.keras.models.load_model('VGG19/CNN_VGG19.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('VGG19/ML_VGG19.pkl')
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
    
    CNN_model = tf.keras.models.load_model('Xception/CNN_Xception.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('Xception/ML_Xception.pkl')
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
    
    CNN_model = tf.keras.models.load_model('EfficientNetV2B3/CNN_EfficientNetV2B3.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('EfficientNetV2B3/ML_EfficientNetV2B3.pkl')
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
    
    CNN_model = tf.keras.models.load_model('EfficientNetV2B3/CNN_EfficientNetV2B3.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('EfficientNetV2B3/ML_EfficientNetV2B3.pkl')
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
    
    CNN_model = tf.keras.models.load_model('DenseNet121/CNN_DenseNet121.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('DenseNet121/ML_DenseNet121.pkl')
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
    
    CNN_model = tf.keras.models.load_model('ResNet50/CNN_ResNet50.h5')
    features = CNN_model.predict(image)
    features = features.reshape(features.shape[0], -1)
    
    ML_model = joblib.load('ResNet50/ML_ResNet50.pkl')
    prediction_RF = ML_model.predict(features)
    return prediction_RF


if __name__ == "__main__":
    main()