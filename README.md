# Smart Sorting System 

This project is a Deep Learning-based Smart Sorting System that classifies fruits and vegetables as *Fresh* or *Rotten* using image recognition. The model is trained using a custom dataset and served using a Flask web application.


##  Project Structure

smart_sort_project
train_model.py
split_dataset.py
app.py
healthy_vs_rotten.h5
static
templates
dataset
dataset/train
dataset/validation


## Dataset Used 
*Fruit and Vegetable Diseases Dataset*  
Source: [Kaggle] (https://www.kaggle.com/datasets/)

## 🧠 Model Info

-> Framework: TensorFlow & Keras 
 -> Model: Convolutional Neural Network (CNN)
 -> Classes: 28 total (e.g., apple_fresh, apple_rotten, banana_fresh, etc.)
 -> Accuracy: High for clean images
