# 🌱 Tomato Disease Detection & AI Crop Advisory System

## 📌 Overview

Tomato Disease Detection & AI Crop Advisory System is a Deep Learning-powered application that identifies diseases in tomato plant leaves from images and provides AI-generated crop advisory recommendations.

The project uses **Transfer Learning with ResNet50**, **PyTorch**, and **Generative AI** to assist farmers, researchers, and agriculture enthusiasts in early disease detection and decision-making.

## 🚀 Features

✅ Upload tomato leaf images for disease prediction
✅ Deep Learning-based disease classification
✅ Transfer Learning using ResNet50
✅ AI-generated crop advisory
✅ Disease description and causes
✅ Preventive measures and treatment suggestions
✅ Interactive Gradio web interface
✅ Deployable on Hugging Face Spaces

## 🧠 Problem Statement

Early identification of plant diseases is crucial for improving crop yield and preventing large-scale agricultural losses.
Traditional disease detection methods often require expert inspection, making the process time-consuming and expensive. This project automates disease detection using Computer Vision and Deep Learning while providing intelligent crop recommendations through Generative AI.

## 🛠️ Tech Stack
### Programming Language
- Python
### Deep Learning Framework
- PyTorch
- Torchvision
### Model Architecture
- ResNet18 (Transfer Learning)
### Data Analysis & Visualization
- NumPy
- Pandas
- Matplotlib
- Seaborn
### Deployment
- Gradio
- Hugging Face Spaces
### AI Integration
- Generative AI
- Prompt Engineering

- ### Dataset Information

| Feature | Value |
|----------|----------|
| Total Images | ~16,012 |
| Number of Classes | 10 |
| Domain | Agriculture / Computer Vision |

### Sample Classes

- Tomato Healthy
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Yellow Leaf Curl Virus

- ## 🏗️ Model Architecture

```python
ResNet50
   │
   ├── Pretrained Weights
   │
   ├── Feature Extraction Layers
   │
   └── Custom Classification Head
            │
            └── 10 Disease Classes
```
