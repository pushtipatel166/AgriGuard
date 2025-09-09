"""
Optimized training script for plant disease detection
Uses data augmentation and efficient model architecture for better performance with smaller datasets
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_optimized_model(num_classes, input_shape=(224, 224, 3)):
    """
    Create an optimized model using EfficientNetB0 for better performance with fewer images
    """
    # Use EfficientNetB0 which is more efficient than MobileNetV2
    base_model = EfficientNetB0(
        weights="imagenet", 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation="softmax"),
    ])
    
    # Use Adam optimizer with learning rate scheduling
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy", "top_3_accuracy"]
    )
    
    return model

def get_optimized_data_generators(dataset_dir, img_size=(224, 224), batch_size=32, val_split=0.2):
    """
    Create optimized data generators with extensive augmentation
    """
    # Training data generator with extensive augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.7, 1.3),
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        channel_shift_range=0.1,
        fill_mode='nearest',
        validation_split=val_split,
    )
    
    # Validation data generator (minimal augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split,
    )
    
    # Training generator
    train_gen = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        subset="training",
        shuffle=True,
        class_mode="categorical",
    )
    
    # Validation generator
    val_gen = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        subset="validation",
        shuffle=False,
        class_mode="categorical",
    )
    
    return train_gen, val_gen

def train_optimized_model():
    """
    Train the optimized model with callbacks and fine-tuning
    """
    # Configuration
    DATASET_DIR = "dataset"
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' not found!")
        return
    
    # Get class information
    train_gen, val_gen = get_optimized_data_generators(DATASET_DIR, IMG_SIZE, BATCH_SIZE)
    num_classes = len(train_gen.class_indices)
    
    print(f"Found {num_classes} classes:")
    for class_name, class_idx in train_gen.class_indices.items():
        print(f"  {class_idx}: {class_name}")
    
    # Save class indices
    with open("class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=4)
    
    # Create model
    model = create_optimized_model(num_classes, IMG_SIZE + (3,))
    print(f"\nModel created with {model.count_params():,} parameters")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'plant_model_optimized.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Phase 1: Train with frozen base model
    print("\n=== Phase 1: Training with frozen base model ===")
    history1 = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen base model
    print("\n=== Phase 2: Fine-tuning with unfrozen base model ===")
    model.layers[0].trainable = True
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy", "top_3_accuracy"]
    )
    
    # Update callbacks for fine-tuning
    callbacks[2] = ModelCheckpoint(
        'plant_model_optimized.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    history2 = model.fit(
        train_gen,
        epochs=20,  # Fewer epochs for fine-tuning
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save("plant_model.h5")
    print("\nModel saved as 'plant_model.h5'")
    
    # Plot training history
    plot_training_history(history1, history2)
    
    # Evaluate model
    evaluate_model(model, val_gen)
    
    return model

def plot_training_history(history1, history2):
    """
    Plot training history for both phases
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Combine histories
    epochs1 = len(history1.history['loss'])
    epochs2 = len(history2.history['loss'])
    total_epochs = epochs1 + epochs2
    
    # Plot accuracy
    axes[0, 0].plot(range(1, epochs1 + 1), history1.history['accuracy'], 'b-', label='Training (Phase 1)')
    axes[0, 0].plot(range(1, epochs1 + 1), history1.history['val_accuracy'], 'r-', label='Validation (Phase 1)')
    axes[0, 0].plot(range(epochs1 + 1, total_epochs + 1), history2.history['accuracy'], 'g-', label='Training (Phase 2)')
    axes[0, 0].plot(range(epochs1 + 1, total_epochs + 1), history2.history['val_accuracy'], 'orange', label='Validation (Phase 2)')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot loss
    axes[0, 1].plot(range(1, epochs1 + 1), history1.history['loss'], 'b-', label='Training (Phase 1)')
    axes[0, 1].plot(range(1, epochs1 + 1), history1.history['val_loss'], 'r-', label='Validation (Phase 1)')
    axes[0, 1].plot(range(epochs1 + 1, total_epochs + 1), history2.history['loss'], 'g-', label='Training (Phase 2)')
    axes[0, 1].plot(range(epochs1 + 1, total_epochs + 1), history2.history['val_loss'], 'orange', label='Validation (Phase 2)')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate
    axes[1, 0].plot(range(1, epochs1 + 1), history1.history.get('lr', [0.001] * epochs1), 'b-', label='Phase 1')
    axes[1, 0].plot(range(epochs1 + 1, total_epochs + 1), history2.history.get('lr', [0.0001] * epochs2), 'g-', label='Phase 2')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # Plot top-3 accuracy
    if 'top_3_accuracy' in history1.history:
        axes[1, 1].plot(range(1, epochs1 + 1), history1.history['top_3_accuracy'], 'b-', label='Training (Phase 1)')
        axes[1, 1].plot(range(1, epochs1 + 1), history1.history['val_top_3_accuracy'], 'r-', label='Validation (Phase 1)')
        axes[1, 1].plot(range(epochs1 + 1, total_epochs + 1), history2.history['top_3_accuracy'], 'g-', label='Training (Phase 2)')
        axes[1, 1].plot(range(epochs1 + 1, total_epochs + 1), history2.history['val_top_3_accuracy'], 'orange', label='Validation (Phase 2)')
        axes[1, 1].set_title('Top-3 Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Top-3 Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, val_gen):
    """
    Evaluate the trained model
    """
    print("\n=== Model Evaluation ===")
    
    # Get predictions
    val_gen.reset()
    predictions = model.predict(val_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_gen.classes
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Top-3 accuracy
    top3_accuracy = 0
    for i in range(len(predictions)):
        top3_preds = np.argsort(predictions[i])[-3:]
        if true_classes[i] in top3_preds:
            top3_accuracy += 1
    top3_accuracy /= len(predictions)
    print(f"Top-3 Accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
    
    # Class-wise accuracy
    print("\nClass-wise Accuracy:")
    class_names = list(val_gen.class_indices.keys())
    for i, class_name in enumerate(class_names):
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
            print(f"  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")

if __name__ == "__main__":
    print("🌱 Starting optimized plant disease detection model training...")
    print("This will create a more efficient model with better performance on smaller datasets.")
    
    # Check GPU availability
    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU detected - training will be faster!")
    else:
        print("⚠️  No GPU detected - training will use CPU (slower)")
    
    # Train the model
    model = train_optimized_model()
    
    print("\n✅ Training completed!")
    print("The optimized model has been saved as 'plant_model.h5'")
    print("You can now use this model with the Flask application.")
