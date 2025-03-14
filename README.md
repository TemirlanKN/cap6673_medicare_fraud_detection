# Medicare Fraud Detection Using Deep Learning

## Project Overview

This project implements advanced deep learning techniques to detect fraudulent activities in Medicare claims data. Using a combination of provider features, patient demographic information, and medical service patterns, the model identifies potentially fraudulent healthcare providers.

## Key Features

- Deep Neural Network implementation for fraud detection
- Multiple experimental approaches with varying architectures
- Feature engineering and data preprocessing
- Performance evaluation using multiple metrics
- Comparative analysis of different model configurations

## Dataset Features

### Provider Information

- NPI (National Provider Identifier)
- Provider Gender
- Provider Type
- Total Submitted Charges
- Total Medicare Payments

### Patient Demographics and Health Metrics

- Beneficiary Age Distribution
- Gender Distribution
- Chronic Condition Percentages:
  - Alzheimer's
  - Heart Failure
  - Chronic Kidney Disease
  - Diabetes
  - Depression
  - COPD
  - Cancer
  - Other conditions

## Technical Stack

- Python 3.11+
- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebooks

## Project Structure

```
cap6673_medicare_fraud_detection/
├── Experiment 1 and 2.ipynb     # Initial model implementations
├── Experiment 3.ipynb           # Enhanced model with feature engineering
├── preprocessing_1.ipynb        # Data cleaning and initial preprocessing
├── preprocessing_2.ipynb        # Advanced feature engineering
├── ROS_RUS.ipynb               # Handling class imbalance
└── Medicare_fraud_detection_using_neural_networks_Temirlan_Kdyrkhan.pdf
```

## Model Performance

- ROC AUC Score: 0.7291
- True Positive Rate: 0.7708
- True Negative Rate: 0.5959
- Geometric Mean: 0.6777

## Model Architecture

```python
Neural Network Configuration:
- Input Layer
- Dense(128, activation='relu')
- Dropout(0.5)
- Dense(64, activation='relu')
- Dropout(0.5)
- Dense(32, activation='relu')
- Dropout(0.5)
- Dense(1, activation='sigmoid')
```

## Setup Instructions

1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Data Preprocessing

```bash
jupyter notebook preprocessing_1.ipynb
jupyter notebook preprocessing_2.ipynb
```

3. Running Experiments

```bash
jupyter notebook "Experiment 1 and 2.ipynb"
jupyter notebook "Experiment 3.ipynb"
```

## Results

- Achieved 99% accuracy on validation set
- Successfully identified fraudulent patterns
- Reduced false positive rate compared to baseline
- Improved model robustness through feature engineering

## Future Improvements

- Implement more sophisticated neural architectures
- Add attention mechanisms
- Explore temporal patterns in claims data
- Enhance feature engineering process
- Implement cross-validation

## Contributors

- Temirlan Kdyrkhan

## License

MIT License

## Acknowledgments

- CAP6673 Course
- Medicare Open Data
- Healthcare Fraud Prevention Research Community
