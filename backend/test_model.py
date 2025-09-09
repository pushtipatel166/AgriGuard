"""
Test script for the plant disease detection model
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
import io

def test_model_local(image_path, model_path="plant_model.h5"):
    """
    Test the model locally with an image
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return
    
    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found!")
        return
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load class indices
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    class_map = {v: k for k, v in class_indices.items()}
    
    # Get input size
    input_shape = model.input_shape
    input_size = (input_shape[1], input_shape[2])
    
    print(f"Model input size: {input_size}")
    print(f"Number of classes: {len(class_map)}")
    
    # Load and preprocess image
    img = image.load_img(image_path, target_size=input_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)[0]
    top_idx = int(np.argmax(predictions))
    confidence = float(predictions[top_idx])
    
    # Get top 3 predictions
    top3_indices = np.argsort(predictions)[-3:][::-1]
    
    print(f"\nPrediction Results:")
    print(f"Top prediction: {class_map[top_idx]} (confidence: {confidence:.4f})")
    print(f"\nTop 3 predictions:")
    for i, idx in enumerate(top3_indices):
        print(f"  {i+1}. {class_map[idx]}: {predictions[idx]:.4f}")

def test_api(image_path, api_url="http://localhost:5000"):
    """
    Test the API with an image
    """
    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found!")
        return
    
    # Test API status
    try:
        response = requests.get(f"{api_url}/")
        print("API Status:", response.json())
    except Exception as e:
        print(f"API not accessible: {e}")
        return
    
    # Test prediction
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nAPI Prediction:")
            print(f"Disease: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
        else:
            print(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error testing API: {e}")

def test_chatbot(api_url="http://localhost:5000"):
    """
    Test the chatbot functionality
    """
    test_messages = [
        "Hello, how are you?",
        "What are common tomato diseases?",
        "How do I treat plant diseases?",
        "What fertilizer should I use?"
    ]
    
    for message in test_messages:
        try:
            response = requests.post(
                f"{api_url}/chat",
                json={"message": message}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\nQ: {message}")
                print(f"A: {result['reply']}")
            else:
                print(f"Chatbot Error: {response.status_code}")
        except Exception as e:
            print(f"Error testing chatbot: {e}")

if __name__ == "__main__":
    print("🌱 Plant Disease Detection Model Tester")
    print("=" * 50)
    
    # Test with a sample image if available
    sample_images = [
        "sample_leaf.jpg",
        "test_image.jpg",
        "leaf.jpg"
    ]
    
    test_image = None
    for img in sample_images:
        if os.path.exists(img):
            test_image = img
            break
    
    if test_image:
        print(f"\nTesting with image: {test_image}")
        print("\n1. Local Model Test:")
        test_model_local(test_image)
        
        print("\n2. API Test:")
        test_api(test_image)
    else:
        print("No test images found. Please add a sample image to test.")
        print("Looking for: sample_leaf.jpg, test_image.jpg, or leaf.jpg")
    
    print("\n3. Chatbot Test:")
    test_chatbot()
    
    print("\n✅ Testing completed!")
