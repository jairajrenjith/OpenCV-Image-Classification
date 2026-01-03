# Simple Image Classification using OpenCV and Machine Learning

This project presents a complete beginner-level implementation of an image classification system using **OpenCV** for image preprocessing and enhancement, and a **classical machine learning algorithm (Support Vector Machine)** for classification.  
The entire project is implemented in a **Jupyter Notebook (`.ipynb`)** to allow step-by-step execution, visualization of results, and easier understanding for beginners.

---

## Objective

The main objective of this project is to:
- Understand how digital images are processed using OpenCV
- Learn basic image preprocessing techniques such as grayscale conversion, resizing, and normalization
- Apply image enhancement methods like brightness adjustment and filtering
- Train a traditional machine learning model on image data
- Evaluate and visualize classification results

This project focuses on **learning fundamentals**, not on achieving very high accuracy.

---

## Dataset Used

The **CIFAR-10** dataset is used in this project.

Dataset details:
- 60,000 color images
- Image size: 32×32 pixels
- 10 different object classes

Classes included:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

For simplicity and faster training, only a **subset of the dataset** is used.

---

## Technologies and Libraries

The following tools and libraries are used:

- **Python** – Programming language
- **OpenCV** – Image preprocessing and enhancement
- **NumPy** – Numerical operations and array handling
- **Matplotlib** – Visualization of images and results
- **Scikit-learn** – Machine learning model (SVM), accuracy, confusion matrix
- **TensorFlow** – Used only to load the CIFAR-10 dataset

---


## Methodology and Steps Followed

### 1. Image Loading
- The CIFAR-10 dataset is loaded using TensorFlow utilities
- A smaller subset of the dataset is selected for faster execution

---

### 2. Image Preprocessing (OpenCV)
Each image undergoes the following preprocessing steps:
- Conversion from RGB to grayscale
- Resizing to a fixed dimension
- Normalization of pixel values to the range 0–1

These steps make the images suitable for traditional machine learning models.

---

### 3. Image Enhancement
To better understand image manipulation using OpenCV:
- Brightness and contrast are adjusted
- Gaussian blur is applied for smoothing
- Original and enhanced images are displayed side-by-side for comparison

---

### 4. Feature Preparation
- Preprocessed images are flattened into one-dimensional feature vectors
- These vectors are used as input to the machine learning model

---

### 5. Model Development
- A **Support Vector Machine (SVM)** classifier is trained using the processed image data
- A linear kernel is used for simplicity and clarity

---

### 6. Model Evaluation
The trained model is evaluated using:
- Accuracy score
- Confusion matrix
- Visualization of test images with predicted and actual labels

This helps in understanding how well the model performs and where it makes mistakes.

---

## Project Structure

```
opencv-image-classification/
│
├── image_classification.ipynb
├── README.md
└── requirements.txt
```

- `image_classification.ipynb` : Main Jupyter Notebook containing the full implementation
- `requirements.txt` : List of required Python libraries
- `README.md` : Project documentation

---

## How to Run the Project

Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

Launch the Jupyter Notebook:
  ```bash
  jupyter notebook image_classification.ipynb
  ```

Run the notebook cells sequentially to observe preprocessing steps, model training, and results.

---


## Results

The model achieves **reasonable accuracy** for a classical machine learning approach without using deep learning techniques.  
The results demonstrate that traditional ML models can still perform basic image classification when combined with proper preprocessing.

---

## Notes

- This project is designed strictly for **educational and learning purposes**
- Accuracy is not the primary goal
- The notebook-based approach helps beginners clearly see inputs, outputs, and visualizations
- This project serves as a strong foundation before moving to deep learning models like CNNs

---
By Jairaj R.
