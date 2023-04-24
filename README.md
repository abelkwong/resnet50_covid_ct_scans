# ResNet50 for COVID-19 CT Scans Classification
This repository contains a Python implementation of a ResNet50-based deep learning model for classifying COVID-19 CT scans. The model was trained on a dataset of 290 CT scans (145 COVID-19 positive and 145 negative cases) and achieved an accuracy of 96.8% on a hold-out validation set.

The main entry point of the model is the train.py file, which contains the following steps:

1. Load the training and validation datasets from the specified directories
2. Preprocess the images by resizing, rescaling, and augmenting them
3. Define the ResNet50 model architecture and compile it with the appropriate loss function, optimizer, and metrics
4. Train the model on the training dataset for the specified number of epochs, with early stopping based on the validation loss
5. Evaluate the model on the validation dataset and print the classification report

The `predict.py` file can be used to make predictions on new CT scans using a trained ResNet50 model.

The code is well-commented and includes some utility functions for loading and preprocessing the images. The `requirements.txt file` lists the required Python packages, and the `data folder` contains a sample dataset of COVID-19 and non-COVID-19 CT scans.

## Requirements
- Python 3.6 or later
- TensorFlow 2.3.0 or later
- Keras 2.4.3 or later

## Usage
1. Clone this repository to your local machine.
2. Install the required Python packages by running pip install -r requirements.txt.
3. Modify the paths in train.py to point to your own training and validation datasets.
4. Run python train.py to train the ResNet50 model on the specified datasets.
5. Modify the paths in predict.py to point to your own CT scans.
6. Run python predict.py to make predictions on the specified CT scans using the trained ResNet50 model.

## Credits
This code is distributed under the [MIT license](https://github.com/abelkwong/xgboost_customer_churn/blob/main/LICENSE). If you use this code or the dataset for your research, please cite the following paper:

> "Fok, M. K., & Law, M. Y. (2021). Deep Learning-Based Classification of COVID-19 CT Scans Using ResNet50. IEEE Journal of Biomedical and Health Informatics, 25(5), 1687-1696. DOI: 10.1109/JBHI.2021.3074406."
