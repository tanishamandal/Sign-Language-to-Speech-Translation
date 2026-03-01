# Sign Language to Speech Translation using Machine Learning

> [cite_start]A real-time computer vision and machine learning system designed to recognize hand gestures and translate them into audible speech, bridging the communication gap for individuals with hearing and speech impairments[cite: 50, 51].

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
[cite_start]Communication is a fundamental aspect of human interaction, but navigating a predominantly verbal world presents significant challenges for those with hearing and speech impairments[cite: 47, 48]. [cite_start]This project addresses this barrier by deploying a real-time translation system that leverages computer vision and machine learning technologies to convert sign language into spoken audio[cite: 50, 51].

## Proposed System and Architecture
The translation pipeline is designed to be seamless and instantaneous:
1. [cite_start]**Input Acquisition:** Captures live hand gestures via a standard webcam[cite: 150].
2. [cite_start]**Pre-processing:** Applies image processing techniques, including background subtraction, thresholding, and contour detection, to isolate the hand[cite: 151].
3. [cite_start]**Classification:** Feeds the processed frames into a trained Convolutional Neural Network (CNN) to classify the specific sign language gesture[cite: 152].

## Key Features
* [cite_start]**Real-Time Translation:** Instantly processes captured gestures and outputs the corresponding speech with minimal latency[cite: 155].
* [cite_start]**Speech Integration:** Utilizes robust Text-to-Speech (TTS) engines, such as Google TTS (gTTS) and pyttsx3, to generate natural, human-like speech[cite: 404, 405].
* [cite_start]**Non-Invasive:** Operates entirely via vision-based input (webcam), eliminating the need for expensive or cumbersome wearable sensors like smart gloves[cite: 158].

## Methodology
* [cite_start]**Algorithm:** The core classification engine is built using Convolutional Neural Networks (CNNs) implemented in TensorFlow[cite: 152, 153].
* [cite_start]**Dataset:** Trained on the Sign Language MNIST dataset sourced from Kaggle[cite: 421, 447]. 
* [cite_start]**Data Pre-processing:** Input images are resized to a standard 28x28 pixel resolution and normalized to ensure consistent model training[cite: 452, 616, 617].
* [cite_start]**Optimization:** The model utilizes the Adam Optimizer paired with a Categorical Cross-Entropy loss function for efficient weight updating and error minimization[cite: 191, 192].

## Performance and Results
Following rigorous testing and evaluation, the selected CNN model achieved the following performance metrics:

| Metric | Achieved Value |
| :--- | :--- |
| **Gesture Classification Accuracy** | [cite_start]93.4% [cite: 636] |
| **Inference Latency** | [cite_start]85 ms (per prediction) [cite: 636] |
| **Optimized Model Size** | [cite_start]18 MB [cite: 636] |

---

## Installation and Setup

Follow these steps to run the project locally on your machine.

### Prerequisites
Ensure you have Python installed. [cite_start]You will also need a functional webcam with a minimum resolution of 400x400 for the real-time capture to function correctly[cite: 608].

### 1. Install Dependencies
Install the required Python libraries by running the following command in your terminal:

```bash
pip install kagglehub opencv-python matplotlib tensorflow pandas numpy scikit-learn seaborn
