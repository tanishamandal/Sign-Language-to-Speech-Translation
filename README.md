Sign Language to Speech Translation using Machine Learning
Project Overview
Communication is a vital aspect of human life, but interacting in a predominantly verbal world poses a significant challenge for individuals with hearing and speech impairments. This project aims to bridge this communication gap by using computer vision and machine learning technologies to recognize hand gestures in real-time and translate them into audible speech.
Proposed System & Architecture
The system captures hand gestures using a webcam and processes them using image pre-processing techniques (background subtraction, thresholding, contour detection). The processed frames are fed into a Convolutional Neural Network (CNN) trained on Sign Language datasets to classify the gesture.
Key Features:
Real-Time Translation: Instantly captures gestures and converts them to speech output.
Speech Integration: Uses Text-to-Speech (TTS) engines like Google TTS (gTTS) or pyttsx3 to generate human-like speech.
Non-Invasive: Operates using standard vision-based input (webcam) without the need for sensor gloves.
Methodology
Algorithm: Convolutional Neural Networks (CNN) using TensorFlow.
Dataset: Sign Language MNIST (via Kaggle). Data was resized to 28x28 and normalized.
Optimization: Adam Optimizer with Categorical Cross-Entropy loss function.
Results
Our chosen CNN model achieved the following performance metrics during evaluation:
Gesture Classification Accuracy: 93.4%.
Latency: 85 ms per prediction.
Model Size: 18 MB.
