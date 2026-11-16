# 🚗 AI-Based Driver Drowsiness Detection and Alert System

An AI-powered system that monitors a driver's alertness in real-time using computer vision and triggers instant alerts when signs of drowsiness are detected — helping prevent road accidents caused by driver fatigue.

## 📌 Problem Statement

Drowsy driving is one of the leading causes of road accidents worldwide. Traditional safety systems fail to detect fatigue-related risks in real time. This project aims to bridge that gap by using AI to continuously monitor driver alertness and issue timely warnings before an accident can occur.

## 🎯 Objective

To design and implement a real-time drowsiness detection system that:
- Continuously monitors the driver's facial features (eyes, yawning, head position)
- Detects signs of drowsiness or fatigue using AI/ML models
- Triggers an immediate audio/visual alert to wake the driver and prevent accidents

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Libraries/Frameworks:** OpenCV, dlib / Mediapipe, NumPy
- **Machine Learning:** CNN / Facial Landmark Detection
- **Alert Mechanism:** Audio alarm / Buzzer trigger
- **Hardware (if applicable):** Webcam / Camera module

## ⚙️ How It Works

1. The camera continuously captures the driver's face in real time.
2. Facial landmarks (especially eyes and mouth) are detected using computer vision techniques.
3. Eye Aspect Ratio (EAR) and yawning patterns are analyzed to determine drowsiness.
4. If drowsiness is detected beyond a threshold duration, an alert (sound/buzzer) is triggered instantly.
5. The system resets and continues monitoring in real time.

## 📂 Project Structure

```
├── dataset/                # Training/testing images (if used)
├── models/                 # Trained model files
├── src/                    # Source code
│   ├── detect.py           # Main detection script
│   ├── alert.py            # Alert/alarm trigger logic
│   └── utils.py            # Helper functions
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- pip

### Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

### Run the Application

```bash
python src/detect.py
```

## ✅ Features

- Real-time facial and eye monitoring
- Accurate drowsiness detection using AI
- Instant alert system to prevent accidents
- Lightweight and easy to deploy

## 🔮 Future Enhancements

- Integration with vehicle hardware for automatic braking/slowdown
- Mobile app support for alerts
- Cloud-based driver monitoring dashboard
- Support for detecting distraction (phone usage, head turns)


