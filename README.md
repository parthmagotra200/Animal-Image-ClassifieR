# 🐾 CNN Animal Image Classifier

A Convolutional Neural Network (CNN) built using TensorFlow/Keras to classify animal images into different categories.

## 🧠 Model Architecture
- 2 Convolutional layers + MaxPooling
- Flatten + Dense layers
- Output layer with softmax for multi-class classification

## 📁 Dataset Structure
```
data/
└── animals/
    ├── cat/
    ├── dog/
    ├── lion/
    ├── elephant/
    └── ... (more classes)
```

You can use any custom dataset with this folder structure.

## 🛠️ How to Run

```bash
pip install -r requirements.txt
python cnn_animal_classifier.py
```

## 📊 Features
- Uses data augmentation and validation split
- Achieves ~92% accuracy on a well-structured 10-class animal dataset
- Saves trained model as `animal_classifier_model.h5`

## 📦 Dependencies
- TensorFlow
- Keras
- (Use `requirements.txt` to install)

---

Built by [Parth Magotra](https://www.linkedin.com/in/parth-magotra-cs)