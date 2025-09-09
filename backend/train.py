import os
import json
from model_utils import build_model, get_data_generators

# Config
DATASET_DIR = os.getenv("DATASET_DIR", "./dataset")
MODEL_PATH = os.getenv("MODEL_PATH", "plant_model.h5")

if __name__ == "__main__":
    # Check dataset directory
    if not os.path.exists(DATASET_DIR):
        raise SystemExit(
            "❌ Please place your dataset in ./backend/dataset with subfolders per class."
        )

    # Load data generators
    train_gen, val_gen = get_data_generators(DATASET_DIR, batch_size=16)
    num_classes = train_gen.num_classes

    # Build and train model
    model = build_model(num_classes)
    model.fit(train_gen, validation_data=val_gen, epochs=10)

    # Save model
    model.save(MODEL_PATH)
    print(f"✅ Saved model to {MODEL_PATH}")

    # Save class indices for inference
    with open("class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=4)

    print("✅ Saved class indices to class_indices.json")
