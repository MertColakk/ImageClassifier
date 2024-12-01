import os
import tensorflow as tf
from keras import layers, models, utils, callbacks
import numpy as np
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self,pretrained:bool, data_dir=None) -> None:
        self.class_names = ["Cat","Dog"]
        
        if not pretrained:
            # Variables
            self.data_dir = data_dir
            
            # is Model Trained?
            self.isTrained = False
            
            # Dataset Directories
            self.train_dataset = tf.keras.utils.image_dataset_from_directory(
                os.path.join(self.data_dir, "train"),
                image_size=(224, 224),
                batch_size=64
            )
            
            self.validation_dataset = tf.keras.utils.image_dataset_from_directory(
                os.path.join(self.data_dir, "validation"),
                image_size=(224, 224),
                batch_size=64
            )
            
            # Normalize Data
            self.normalization_layer = layers.Rescaling(1./255)
            
            self.train_dataset = self.train_dataset.map(lambda x, y: (self.normalization_layer(x), y))
            self.validation_dataset = self.validation_dataset.map(lambda x, y: (self.normalization_layer(x), y)) 
            
            # Create Model
            self.model = models.Sequential([
                layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(64,(3,3),activation='relu'),
                layers.MaxPooling2D(2,2),
                layers.Conv2D(128,(3,3),activation='relu'),
                layers.MaxPooling2D(2,2),
                
                # Fully Connected Layers
                layers.Flatten(),
                layers.Dense(128,activation='relu'),
                layers.Dense(2,activation='softmax')
            ])
            
            # Compile Model
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )  
        else:
            self.model = models.load_model(data_dir)
            
    def process_image(self,image_path):
        img = utils.load_img(image_path, target_size = (224,224))
        img_array = utils.img_to_array(img)
        img_array = tf.expand_dims(img_array,axis=0)
        img_array = img_array / 255.0
        
        return img_array
    
    def predict(self,image_path):
        prediction = self.model.predict(self.process_image(image_path))
        
        predicted_class = self.class_names[np.argmax(prediction)]
        confidence = np.max(prediction)  
        
        return f"Class: {predicted_class}, Confidence: {confidence:.2f}"
        
    def train(self,epoch_size:int):
        if not self.isTrained:
            callback = callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
            
            history = self.model.fit(
                self.train_dataset,
                validation_data = self.validation_dataset,
                epochs = epoch_size,
                callbacks=[callback]
            )
            
            self.isTrained = True
            
            self.visualize_train(history)
            
            self.save_model()
            
    def visualize_train(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Train Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Loss')
        plt.show()
        
    def save_model(self):
        if self.isTrained:
            self.model.save('cat_and_dog_classifier.h5')
            
        
        
        
        
        
        
