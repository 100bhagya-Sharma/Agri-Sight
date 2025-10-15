import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class DiseaseClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=38, checkpoint_dir='checkpoints'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.model = self.build_model()
        self.compile_model()
        
    def build_model(self):
        """Build a simpler CNN model with L2 regularization"""
        l2_reg = 0.001  # L2 regularization factor
        
        model = models.Sequential([
            layers.Rescaling(1./255, input_shape=self.input_shape),
            # First convolutional block
            layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, kernel_regularizer=regularizers.l2(l2_reg)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self):
        """Compile the model with Adam optimizer"""
        initial_learning_rate = 0.0001
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, train_data, validation_data, epochs=50, batch_size=32):
        """Train the model with callbacks"""
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Callbacks
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint
            ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_generator):
        """Evaluate the model with classification report and confusion matrix"""
        test_generator.reset()
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names)
        conf_matrix = confusion_matrix(y_true, y_pred)
        return report, conf_matrix
    
    def predict(self, data):
        """Make predictions on new data"""
        return self.model.predict(data)
    
    def save(self, filepath):
        """Save the model to disk"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load a saved model from disk"""
        self.model = tf.keras.models.load_model(filepath)