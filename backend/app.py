import os
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import openai

# Flask setup
app = Flask(__name__)
CORS(app)

# Config
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
MODEL_PATH = "plant_model.h5"
model = None
class_map = None
input_size = (224, 224)  # Default input size

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    # Load class indices from JSON file
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    # Create reverse mapping (index to class name)
    class_map = {v: k for k, v in class_indices.items()}
    
    # Try to determine input size from model
    try:
        input_shape = model.input_shape
        if input_shape and len(input_shape) >= 3:
            input_size = (input_shape[1], input_shape[2])
    except:
        pass
    
    print(f"Model loaded successfully with {len(class_map)} classes")
    print(f"Input size: {input_size}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train the model first using train.py or train_optimized.py")

# OpenAI API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY
else:
    print("Warning: OPENAI_API_KEY not set. Chatbot will use fallback responses.")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or class_map is None:
        return jsonify({"error": "Model or class map not found. Train the model first."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)

    # Preprocess image with dynamic size
    img = image.load_img(path, target_size=input_size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    top_idx = int(np.argmax(preds))
    label = class_map.get(top_idx, "Unknown")
    confidence = float(preds[top_idx])

    # Clean up the uploaded file
    try:
        os.remove(path)
    except:
        pass

    return jsonify({"prediction": label, "confidence": round(confidence, 4)})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_msg = data["message"]

    if not OPENAI_KEY:
        # Enhanced fallback responses with better keyword matching
        user_msg_lower = user_msg.lower()
        
        # Disease-specific responses
        if any(word in user_msg_lower for word in ["late blight", "late_blight", "tomato late blight"]):
            return jsonify({"reply": "Tomato Late Blight is a serious fungal disease caused by Phytophthora infestans. Symptoms include dark, water-soaked lesions on leaves, stems, and fruits. To treat: 1) Remove infected plant parts immediately, 2) Apply copper-based fungicides, 3) Improve air circulation, 4) Avoid overhead watering, 5) Use resistant varieties. Prevention is key - ensure proper spacing and avoid wet conditions."})
        
        elif any(word in user_msg_lower for word in ["early blight", "early_blight"]):
            return jsonify({"reply": "Early Blight is a common fungal disease affecting tomatoes and potatoes. Symptoms include brown spots with concentric rings on leaves. Treatment: 1) Remove infected leaves, 2) Apply fungicides containing chlorothalonil or copper, 3) Mulch around plants, 4) Water at soil level, 5) Rotate crops annually. The disease spreads in warm, humid conditions."})
        
        elif any(word in user_msg_lower for word in ["bacterial spot", "bacterial_spot"]):
            return jsonify({"reply": "Bacterial Spot is caused by Xanthomonas bacteria. Symptoms include small, dark spots on leaves and fruits. Treatment: 1) Remove infected plants, 2) Apply copper-based bactericides, 3) Avoid overhead watering, 4) Use disease-free seeds, 5) Practice crop rotation. This disease is difficult to control once established, so prevention is crucial."})
        
        elif any(word in user_msg_lower for word in ["mosaic virus", "mosaic_virus"]):
            return jsonify({"reply": "Mosaic Virus causes mottled yellow and green patterns on leaves, stunted growth, and deformed fruits. There's no cure for viral diseases. Prevention: 1) Use virus-free seeds, 2) Control aphids (vectors), 3) Remove infected plants immediately, 4) Disinfect tools, 5) Plant resistant varieties. Once infected, plants should be destroyed to prevent spread."})
        
        elif any(word in user_msg_lower for word in ["yellow leaf curl", "yellow_leaf_curl"]):
            return jsonify({"reply": "Tomato Yellow Leaf Curl Virus causes upward curling of leaves, yellowing, and stunted growth. It's spread by whiteflies. Treatment: 1) Control whitefly populations with insecticides, 2) Remove infected plants, 3) Use reflective mulches, 4) Plant resistant varieties, 5) Practice good sanitation. This virus can devastate entire crops if not managed properly."})
        
        elif any(word in user_msg_lower for word in ["spider mites", "spider_mites"]):
            return jsonify({"reply": "Spider mites are tiny pests that cause yellow stippling on leaves and fine webbing. Treatment: 1) Spray with water to dislodge mites, 2) Apply insecticidal soap or neem oil, 3) Introduce beneficial insects like ladybugs, 4) Increase humidity, 5) Remove heavily infested leaves. They thrive in hot, dry conditions."})
        
        elif any(word in user_msg_lower for word in ["leaf mold", "leaf_mold"]):
            return jsonify({"reply": "Leaf Mold is a fungal disease that causes yellow spots on upper leaf surfaces and fuzzy growth underneath. Treatment: 1) Improve air circulation, 2) Reduce humidity, 3) Apply fungicides, 4) Remove infected leaves, 5) Avoid overhead watering. This disease is common in greenhouse environments with high humidity."})
        
        elif any(word in user_msg_lower for word in ["septoria", "septoria_leaf_spot"]):
            return jsonify({"reply": "Septoria Leaf Spot causes small, circular spots with dark borders on leaves. Treatment: 1) Remove infected leaves, 2) Apply fungicides, 3) Mulch around plants, 4) Water at soil level, 5) Practice crop rotation. The disease spreads through splashing water and infected plant debris."})
        
        elif any(word in user_msg_lower for word in ["target spot", "target_spot"]):
            return jsonify({"reply": "Target Spot causes circular lesions with concentric rings on leaves and fruits. Treatment: 1) Remove infected plant parts, 2) Apply fungicides, 3) Improve air circulation, 4) Avoid overhead watering, 5) Use resistant varieties. This disease is more common in warm, humid climates."})
        
        # General disease questions
        elif any(word in user_msg_lower for word in ["disease", "sick", "infected", "problem", "issue"]):
            return jsonify({"reply": "I can help identify plant diseases! For accurate diagnosis, please upload an image using the disease detection feature. Common plant diseases include fungal infections (blights, spots, molds), bacterial diseases, and viral infections. Each requires different treatment approaches. Prevention through proper care, good sanitation, and resistant varieties is always the best strategy."})
        
        # Fertilizer questions
        elif any(word in user_msg_lower for word in ["fertilizer", "nutrient", "feeding", "fertilize"]):
            return jsonify({"reply": "For healthy plants, use balanced fertilizers with NPK ratios appropriate for your crop. Organic options include compost, manure, fish emulsion, and bone meal. Apply fertilizers according to plant needs - more nitrogen for leafy growth, more phosphorus for flowering/fruiting. Always follow package instructions and avoid over-fertilizing, which can burn roots."})
        
        # Watering questions
        elif any(word in user_msg_lower for word in ["water", "watering", "irrigation", "moisture"]):
            return jsonify({"reply": "Proper watering is crucial for plant health. Water deeply but infrequently to encourage deep root growth. Check soil moisture by inserting your finger 1-2 inches deep. Water when the top inch is dry. Avoid overhead watering to prevent disease. Early morning is the best time to water. Ensure good drainage to prevent root rot."})
        
        # Pest questions
        elif any(word in user_msg_lower for word in ["pest", "insect", "bug", "aphid", "whitefly"]):
            return jsonify({"reply": "For pest control, start with natural methods: 1) Hand-pick larger pests, 2) Use insecticidal soap or neem oil, 3) Introduce beneficial insects, 4) Remove affected plant parts, 5) Keep garden clean. Chemical pesticides should be a last resort. Identify the specific pest for targeted treatment. Prevention through healthy plants and good garden hygiene is most effective."})
        
        # General plant care
        elif any(word in user_msg_lower for word in ["care", "grow", "planting", "garden", "crop"]):
            return jsonify({"reply": "Successful plant care involves: 1) Choosing the right location with adequate sunlight, 2) Preparing soil with good drainage and nutrients, 3) Proper spacing for air circulation, 4) Regular watering and fertilizing, 5) Monitoring for pests and diseases, 6) Timely harvesting. Each plant has specific needs, so research your particular crop for best results."})
        
        # Greeting responses
        elif any(word in user_msg_lower for word in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return jsonify({"reply": "Hello! I'm your plant disease expert assistant. I can help you identify plant diseases, provide care advice, and answer questions about fertilizers, watering, and pest control. For disease diagnosis, upload an image using the detection feature. What would you like to know about plant care?"})
        
        # Default response
        else:
            return jsonify({"reply": "I'm here to help with plant care and disease identification! You can ask me about specific diseases, plant care, fertilizers, watering, or pest control. For accurate disease diagnosis, please upload an image using the disease detection feature. What specific question do you have about your plants?"})

    system_prompt = (
        "You are an expert agricultural assistant. Provide concise, practical advice about crop care, "
        "disease management, fertilization, and irrigation. "
        "If the user asks for diagnosis from an image, remind them to use the image upload feature "
        "and provide general next steps."
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=300,
        )

        reply = resp["choices"][0]["message"]["content"]
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return jsonify({"reply": "Sorry, I'm having trouble connecting to the AI service. Please try again later."})


@app.route("/")
def root():
    return jsonify({
        "message": "Plant Disease Detector API is running!", 
        "endpoints": ["/predict", "/chat", "/model-info"],
        "model_loaded": model is not None,
        "classes": len(class_map) if class_map else 0,
        "input_size": input_size
    })

@app.route("/model-info")
def model_info():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_loaded": True,
        "input_size": input_size,
        "num_classes": len(class_map),
        "classes": list(class_map.values()),
        "model_summary": str(model.count_params()) + " parameters"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
