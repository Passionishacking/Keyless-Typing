✋ Hand Gesture Recognition System (Machine Learning)

A real-time Hand Gesture Recognition System built using Python, OpenCV, MediaPipe, and Machine Learning.
The system recognizes hand gestures (A–Z / predefined gestures) from a webcam and converts them into meaningful actions or text.

🚀 Features

🎥 Real-time hand detection using webcam

✋ Hand landmark extraction with MediaPipe

🧠 Machine Learning–based gesture classification

⌨️ Virtual typing support (A–Z, Space, Delete, Enter)

⚡ Low-latency and lightweight model

🖥️ Simple and interactive UI for live prediction

🛠️ Tech Stack
Core Technologies

Python 3.10+

OpenCV

MediaPipe



Scikit-learn / TensorFlow (as used)

ML / AI

Hand landmark detection (21 key points)

Feature-based classification

Supervised learning model

📂 Project Structure
hand-gesture-recognition/
│
├── dataset/                    # Collected hand landmark data
├── model/                      # Trained ML model
├── analysis_outputs/           # Training graphs & analysis
│
├── collect_landmarks.py        # Dataset creation
├── train_classifier.py         # Model training
├── live_landmark_infer.py      # Real-time inference
├── hand_gesture_landmark_ui.py # UI for gesture recognition
├── analyze_model.py            # Model evaluation
├── plot_training_curve.py      # Accuracy/Loss plots
│
├── requirements.txt
├── .gitignore
└── README.md

🧠 How It Works

1️⃣ Webcam Input

Captures live video using OpenCV

2️⃣ Hand Detection

MediaPipe detects hand and extracts 21 landmarks

3️⃣ Feature Extraction

Landmark coordinates converted into feature vectors

4️⃣ Model Prediction

Trained ML model predicts the gesture

5️⃣ Output

Recognized gesture shown as text / action on screen

📊 Dataset

Custom dataset created using hand landmarks

Each gesture saved as numerical landmark data

Stored in CSV format for easy training
