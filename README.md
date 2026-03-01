# Sign Language to Speech Translation using Machine Learning

>A real-time computer vision and machine learning system designed to recognize hand gestures and translate them into audible speech, bridging the communication gap for individuals with hearing and speech impairments.

---

## Table of Contents
* Project Overview
* Proposed System and Architecture
* Key Features
* Methodology
* Performance and Results
* Installation and Setup

---

## Project Overview
]Communication is a fundamental aspect of human interaction, but navigating a predominantly verbal world presents significant challenges for those with hearing and speech impairments.This project addresses this barrier by deploying a real-time translation system that leverages computer vision and machine learning technologies to convert sign language into spoken audio.

## Proposed System and Architecture
The translation pipeline is designed to be seamless and instantaneous:
1.Input Acquisition: Captures live hand gestures via a standard webcam.
2.Pre-processing: Applies image processing techniques, including background subtraction, thresholding, and contour detection, to isolate the hand.
3.Classification:Feeds the processed frames into a trained Convolutional Neural Network (CNN) to classify the specific sign language gesture.

## Key Features
* Real-Time Translation:Instantly processes captured gestures and outputs the corresponding speech with minimal latency.
* Speech Integration:Utilizes robust Text-to-Speech (TTS) engines, such as Google TTS (gTTS) and pyttsx3, to generate natural, human-like speech.
* Non-Invasive: Operates entirely via vision-based input (webcam), eliminating the need for expensive or cumbersome wearable sensors like smart gloves.

## Methodology
* Algorithm:The core classification engine is built using Convolutional Neural Networks (CNNs) implemented in TensorFlow.
* Dataset:Trained on the Sign Language MNIST dataset sourced from Kaggle. 
* Data Pre-processing:Input images are resized to a standard 28x28 pixel resolution and normalized to ensure consistent model training.
* Optimization:The model utilizes the Adam Optimizer paired with a Categorical Cross-Entropy loss function for efficient weight updating and error minimization.

## Installation and Setup

Follow these steps to run the project locally on your machine.

### Prerequisites
Ensure you have Python installed. You will also need a functional webcam with a minimum resolution of 400x400 for the real-time capture to function correctly.

### 1. Install Dependencies
Install the required Python libraries by running the following command in your terminal:

```bash
pip install kagglehub opencv-python matplotlib tensorflow pandas numpy scikit-learn seaborn
