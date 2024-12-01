# Cat and Dog Image Classifier

This repository contains a **Cat and Dog Image Classifier** built using TensorFlow and Keras. The classifier can distinguish between images of cats and dogs with the help of a Convolutional Neural Network (CNN).

## Features
- **Train a new model**: Train a CNN on your custom dataset.
- **Pretrained model support**: Load a pretrained model to make predictions without retraining.
- **Custom image prediction**: Predict the class of a given image (cat or dog) with confidence level.
- **Early stopping**: Prevents overfitting by stopping training when validation loss stops improving.
- **Training visualization**: Visualize the accuracy and loss curves for training and validation datasets.

---

## Requirements
This project runs on Python and uses TensorFlow and Keras libraries. The following are the requirements to set up the environment:

- Python (3.8 or above recommended)
- TensorFlow (2.0 or above)
- Keras
- NumPy
- Matplotlib
- OS and pathlib (default Python libraries)

### Setting up the environment
Use `miniconda` to create a virtual environment and install dependencies:

```bash
# Create a virtual environment
conda create -n image_classifier_env python=3.8 -y

# Activate the environment
conda activate image_classifier_env

# Install the required packages
pip install tensorflow numpy matplotlib
```

---

## Directory Structure
The project expects the dataset to be organized in the following structure:

```
data/
│
├── train/
│   ├── Cat/
│   └── Dog/
│
└── validation/
    ├── Cat/
    └── Dog/
```

- **train/**: Training dataset with `Cat` and `Dog` subdirectories.
- **validation/**: Validation dataset with `Cat` and `Dog` subdirectories.

---

## Code Overview

### 1. **`Classifier` Class**
This is the main class containing methods for:
- Loading datasets and preprocessing images.
- Training and saving the CNN model.
- Making predictions for new images.
- Visualizing training progress.

#### Key Methods
- **`__init__(pretrained: bool, data_dir: str = None)`**  
  Initializes the classifier. If `pretrained=True`, a saved model will be loaded; otherwise, a new model will be created.

- **`process_image(image_path: str)`**  
  Preprocesses an input image to be compatible with the model.

- **`predict(image_path: str)`**  
  Predicts the class (Cat/Dog) and confidence for a given image.

- **`train(epoch_size: int)`**  
  Trains the model on the provided dataset for the specified number of epochs and saves the model.

- **`visualize_train(history)`**  
  Plots training and validation accuracy/loss over epochs.

- **`save_model()`**  
  Saves the trained model to a `.h5` file.

---

## Usage

### 1. **Training a New Model**
```python
from classifier import Classifier

# Initialize the classifier with training data
data_dir = "path_to_dataset"
classifier = Classifier(pretrained=False, data_dir=data_dir)

# Train the model
classifier.train(epoch_size=10)
```

### 2. **Using a Pretrained Model**
```python
from classifier import Classifier

# Load a pretrained model
model_path = "cat_and_dog_classifier.h5"
classifier = Classifier(pretrained=True, data_dir=model_path)

# Make predictions
image_path = "path_to_image.jpg"
result = classifier.predict(image_path)
print(result)  # Output: Class: Cat, Confidence: 0.95
```

---

## Visualizing Training Results
The training process outputs accuracy and loss graphs for both training and validation datasets.

Example output:

- **Accuracy graph**: Shows how the accuracy improves over epochs for training and validation.
- **Loss graph**: Shows how the loss decreases over epochs for training and validation.

---

## Saving and Loading the Model
The model is automatically saved as `cat_and_dog_classifier.h5` after training. This can be loaded later to skip the training process and directly make predictions.

---

## Example Prediction
```python
image_path = "example_image.jpg"
result = classifier.predict(image_path)
print(result)  # Example Output: Class: Dog, Confidence: 0.98
```

---

## Acknowledgments
This project uses TensorFlow and Keras to implement the CNN. The structure of the dataset should follow typical supervised image classification guidelines.

Feel free to fork this repository and modify it for your needs!

---
