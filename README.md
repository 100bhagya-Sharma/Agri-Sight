# Agri-Sight: Advanced Crop Disease Detection

An AI-powered web application for accurate crop disease diagnosis from leaf images, specifically engineered to be robust against real-world conditions.

## Features

- Advanced image preprocessing and augmentation pipeline
- Transfer learning with state-of-the-art CNN architectures
- Robust against poor lighting, shadows, and partial occlusion
- User-friendly web interface for instant disease detection
- High accuracy and reliability in real-world conditions

## Project Structure

```
agrisight/
├── data/
│   ├── raw/            # Original PlantVillage dataset
│   └── processed/      # Augmented images
├── notebooks/
│   └── 1.0-eda-augmentation-strategy.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   └── app.py
├── models/
│   └── agrisight_model.h5
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agrisight.git
cd agrisight
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
preprocessor.prepare_dataset("data/raw", "data/processed")
```

2. Training the Model:
```python
from src.model import DiseaseClassifier

classifier = DiseaseClassifier(num_classes=14)  # Adjust based on your dataset
classifier.train(train_data, validation_data)
```

3. Running the Web Interface:
```bash
python src/app.py
```

## Model Architecture

The project uses transfer learning with three pre-trained models:
- VGG16
- ResNet50
- InceptionV3

The final model is selected based on performance metrics including accuracy, precision, recall, and F1-score.

## Data Augmentation Strategy

The preprocessing pipeline includes:
- Lighting & contrast variations
- Occlusion simulation
- Perspective shifts
- Random geometric transformations

This ensures the model performs well in various real-world conditions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.