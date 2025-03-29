# Aadhaar-OCR
Aadhaar Card (India ID Proof) OCR Tensorflow Lite model for Android Apps

## Introduction
This is a simple OCR model for Aadhaar card using Tensorflow Lite. The model is trained using the Ultralytics YOLOv8. The model is trained on the Aadhaar card dataset which is generated using the Aadhaar card generator script. The model is trained using the TensorFlow Object Detection API. The model is converted to the TensorFlow Lite model for Android Apps.

![](https://github.com/ojasgulati/Aadhaar-OCR/blob/main/images/aadhar_ocr.gif)

# Aadhar Card Object Detection - YOLOv8

This project focuses on detecting and identifying Aadhar Card(India ID Proof)-related objects using YOLOv8, a state-of-the-art object detection model developed by Ultralytics. The solution is optimized for high accuracy and real-time performance, making it ideal for Aadhar document verification and processing applications.

## Data Flow Diagram:
![](https://github.com/ojasgulati/Aadhaar-OCR/blob/main/images/aadhar_ocr_step_process.jpg)

## Key Features:

+ **YOLOv8-based Detection**: Utilizes the latest YOLOv8 model for precise object detection.

+ **Real-time Processing**: Optimized to run efficiently with GPU acceleration.

+ **Custom Training**: Trained specifically on Aadhar Card images to detect key elements such as name, number, and photo.

+ **Pre-trained Models**: Supports fine-tuning with custom datasets.

+ **Integration-ready**: Can be integrated into various applications for document authentication.

## Technologies Used:

+ **Deep Learning Framework**: Ultralytics YOLOv8

+ **Programming Language**: Python

+ **Deployment**: Google Colab / Local GPU

+ **Libraries**: OpenCV, TensorFlow/PyTorch

+ **Mobile Deployment**: TensorFlow Lite (TFLite) for Android

## Getting Started

1. **Install dependencies**: Ensure all required packages are installed.

2. **Prepare Dataset**: Use annotated Aadhar images for training.

3. **Train the Model**: Run the training script to fine-tune YOLOv8.

4. **Convert to TFLite**: Convert the trained model to TensorFlow Lite format for mobile deployment.

5. **Run Inference**: Test the model on sample images.