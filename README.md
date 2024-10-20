---

# **Freshness Detection & Expiry Date Extraction System**  

## **Project Overview**  
This project offers a solution to detect the freshness and quality of fresh produce and extract critical information from product labels using Optical Character Recognition (OCR). It aims to automate the process of monitoring product quality and expiry dates to improve inventory management and reduce food waste.

## **Table of Contents**  
1. [Objective](#objective)  
2. [Technologies Used](#technologies-used)  
3. [Features](#features)  
4. [Dataset](#dataset)  
5. [Model Architecture](#model-architecture)  
6. [Setup and Installation](#setup-and-installation)
8. [Results and Plots](#results-and-plots)  
9. [Impact](#impact)  
10. [Future Scope](#future-scope)

---

## **Objective**  
- Extract essential product details using OCR from labels.  
- Detect expiry dates and calculate the remaining shelf life.  
- Classify fresh and rotten produce with high accuracy using multiple models.  

---

## **Technologies Used**  
- **OCR**: PaddleOCR for extracting text from labels.  
- **Image Classification Models**:  
  - ResNet50, EfficientNetB0, DenseNet121, and InceptionV3 for freshness detection and classification.  
  - TensorFlow/Keras for model training, evaluation, and comparative analysis.  
- **Environment**: Kaggle Notebooks with GPU support for fast training.  
- **Programming Language**: Python  

---

## **Features**  
- **OCR Extraction**: Extract expiry date and label details with confidence scores.  
- **Shelf-life Calculation**: Compute remaining days from extracted expiry dates.  
- **Freshness Detection**: Classify fruits and vegetables as fresh or rotten using state-of-the-art models.  
- **Real-Time Performance**: Process images quickly and provide freshness evaluation and label extraction results.  
- **Multiple Model Comparison**: Evaluate different models to ensure the highest accuracy.  

---

## **Dataset**  
The [kaggle dataset](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification) and contains images of fruits and vegetables classified into two categories:  
- **Fresh Produce**: Includes fresh apples, bananas, tomatoes, etc.  
- **Rotten Produce**: Includes rotten versions of the same fruits/vegetables.

---

## **Model Architecture**  
The project compares the performance of the following models:  
1. **ResNet50**  
2. **EfficientNetB0**  
3. **DenseNet121**  
4. **InceptionV3**  

All models use **ImageNet pre-trained weights**, and only the top layers are fine-tuned with the current dataset. The models are evaluated using metrics such as accuracy, loss, and validation accuracy. 

---

## **Setup and Installation**  

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/your-username/freshness-detection-ocr.git  
   cd freshness-detection-ocr  
   ```

2. **Install Dependencies**:  
   Make sure you have Python installed, then install the necessary libraries.  
   ```bash
   pip install tensorflow paddlepaddle paddleocr matplotlib scikit-learn  
   ```

3. **Download the Dataset**:  
   Place the dataset in the `dataset/Train` and `dataset/Test` directories as required.  

4. **Set up Kaggle GPU (if needed)**:  
   This solution is optimized for training on **Kaggle with GPU** enabled.  

---

## **Results and Plots**  
- **Accuracy**:
  Training Accuracy: 99.43% - Training Loss: 0.0307
  Validation Accuracy: 98.50% - Validation loss: 0.0493
- **OCR Output**: Display extracted text and confidence scores from product labels.

---

## **Impact**  
- **Automation**: Reduces the need for manual inspection of product labels and quality.  
- **Waste Reduction**: Helps prevent spoilage by providing timely insights.  
- **Better Inventory Management**: Enables better tracking of product shelf life and stock rotation.  

---

## **Future Scope**  
- **Integration with IoT Sensors**: Use sensors to detect environmental conditions affecting product quality.  
- **Deployment on Mobile Devices**: Build a mobile app for real-time product quality detection.  
- **Expand Dataset**: Add more products and include regional produce.  
- **API Integration**: Provide APIs for seamless integration with inventory management systems.  

---

## **Contributors**  
- **Aniket Dhage** – AI/ML Engineer, Python Developer, Data Scientist​  
- **Sanika Butle** - Python Developer, Software Engineer​ 

---

## **License**  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  

---
