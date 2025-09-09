# 🌱 Plant Disease Detector

A modern web application that uses AI to detect plant diseases from leaf images and provides expert advice through an integrated chatbot.

## Features

- **🔍 Disease Detection**: Upload leaf images to get instant disease diagnosis
- **💬 Expert Chatbot**: Ask questions about plant care, diseases, fertilizers, and farming
- **🎯 Accurate Results**: AI model trained on thousands of plant images
- **⚡ Fast Analysis**: Get results in seconds
- **📱 Responsive Design**: Works on desktop and mobile devices

## Supported Plant Diseases

The model can detect the following diseases:

### Potato Diseases
- Early Blight
- Late Blight  
- Healthy

### Tomato Diseases
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites (Two-spotted spider mite)
- Target Spot
- Tomato Mosaic Virus
- Tomato Yellow Leaf Curl Virus
- Healthy

## How to Run

### Backend (Flask API)
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:5000`

### Frontend (Web Interface)
1. Open the `frontend/simple.html` file in your web browser
2. Or navigate to the frontend directory and run:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Usage

1. **Disease Detection**:
   - Click on the upload area to select a leaf image
   - Click "Analyze Disease" to get the diagnosis
   - View the disease name and confidence percentage

2. **Expert Chat**:
   - Type your questions about plant care in the chat box
   - Get instant expert advice on diseases, fertilizers, watering, etc.
   - The chatbot works even without OpenAI API key (uses fallback responses)

## API Endpoints

- `GET /` - API status and available endpoints
- `POST /predict` - Upload image for disease detection
- `POST /chat` - Send message to the expert chatbot

## Technical Details

- **Backend**: Flask with TensorFlow/Keras for AI model
- **Frontend**: React with Tailwind CSS for modern UI
- **Model**: MobileNetV2 transfer learning model
- **Image Processing**: 224x224 pixel input size
- **Supported Formats**: JPG, PNG, JPEG

## Model Training

To retrain the model with your own dataset:

1. Place your images in the `backend/dataset/` directory
2. Run the training script:
   ```bash
   cd backend
   python train.py
   ```

## Environment Variables

- `OPENAI_API_KEY`: Optional OpenAI API key for enhanced chatbot responses

## Troubleshooting

- **Model not found**: Make sure `plant_model.h5` exists in the backend directory
- **CORS errors**: The backend includes CORS headers for cross-origin requests
- **Image upload issues**: Ensure images are in supported formats (JPG, PNG, JPEG)

## Contributing

Feel free to contribute by:
- Adding more plant disease classes
- Improving the UI/UX
- Enhancing the chatbot responses
- Optimizing the model for better accuracy

## License

This project is open source and available under the MIT License.
