# Smart Sorting System 

This project is a Deep Learning-based Smart Sorting System that classifies fruits and vegetables as *Fresh* or *Rotten* using image recognition. The model is trained using a custom dataset and served using a Flask web application.


##  Project Structure

smart_sort_project/ 
# Python script to train the CNN model 
├── train_model.py    
├── split_dataset.py         # Script to split dataset into train/test 
├── app.py                   # Flask web app to classify uploaded images 
├── healthy_vs_rotten.h5    
# Trained Keras model 
├── static/                 
# Uploaded images and CSS 
├── templates/            
# HTML files for Flask 
├── dataset/               # Full dataset (not uploaded due to size) │  
  ├── train/ 
  └── validation/

## Dataset Used 
*Fruit and Vegetable Diseases Dataset*  
Source: [Kaggle] (https://www.kaggle.com/datasets/)

## 🧠 Model Info

-> Framework: TensorFlow & Keras
-> Model: Convolutional Neural Network (CNN)
-> Classes: 28 total (e.g., apple_fresh, apple_rotten, banana_fresh, etc.)
-> Accuracy: High for clean images
