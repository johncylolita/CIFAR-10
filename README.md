# CIFAR-10 Image Classification using Deep Learning

## Objective
The objective of this project is to build and evaluate deep learning models for image classification using the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to train models that can correctly classify these images into one of the 10 classes.

## Data Preprocessing
- CIFAR-10 dataset is loaded using the Keras library and divided into training and testing sets.
- Input images are normalized to have pixel values in the range of [0, 1].
- Input labels are one-hot encoded to convert them into categorical vectors.

## Data Exploration
An interactive widget is included to visualize random images from the training data along with their corresponding labels, allowing for visual exploration of the dataset.

## Data Augmentation
Data augmentation is applied to the training data using the Keras data augmentation pipeline. The augmentation includes random horizontal flipping and random rotations up to 36 degrees, which helps in generating new variations of the training data, making the model more robust and less prone to overfitting.

## Model Architectures

**1. make_model1**: This model incorporates data augmentation, convolutional layers with increasing filters (32, 64, 128), batch normalization, activation functions, and dropout layers. It also employs global average pooling before the final classification layer.

**2. make_model2**: This model mirrors `make_model1` but excludes the MaxPooling and dropout layers, aiming to gauge the performance when these layers are absent.

**3. make_model3**: This model eschews data augmentation to analyze performance in its absence.

## Model Training
- Models are compiled using the Adam optimizer with a learning rate of 1e-3 and categorical cross-entropy loss function.
- Training is conducted for 20 epochs with a batch size of 64, using 20% of the training data as the validation set.
- Training time is documented for every model.

## Model Evaluation
Test accuracy is calculated for each model using the test dataset.

## Results

- **Model1** (with data augmentation, dropout, and MaxPooling): 
  - Test accuracy: 52.68%
  - Training time: 300.343 seconds
- **Model2** (without data augmentation, dropout, and MaxPooling):
  - Test accuracy: 70.85%
  - Training time: 277.576 seconds
- **Model3** (without data augmentation):
  - Test accuracy: 68.84%
  - Training time: 570.334 seconds

## Model Performance Analysis
- Model2 achieved the highest test accuracy of 70.85%, suggesting that for this dataset, eliminating the dropout and MaxPooling layers might enhance the model's performance.
- Model1, despite data augmentation, did not perform as well as Model2, indicating that the augmentation strategy might not be optimal for this dataset.
- Model3, which excludes data augmentation, lags behind both Model1 and Model2, reinforcing the significance of data augmentation in elevating model performance.

## Confusion Matrix and Classification Report
For Model1, the code generates a confusion matrix and classification report, offering insights into the model's strengths and weaknesses across various classes.

## Conclusion
In this endeavor, I delved into varying deep learning architectures for image classification on the CIFAR-10 dataset. Notably, Model2, which omitted dropout and MaxPooling layers, surfaced with the most impressive test accuracy of 70.85%. This exercise underscores the pivotal role of data augmentation and architectural selections in determining model efficacy. To potentially achieve superior results, future exploration can encompass alternative model architectures, hyperparameter optimization, and diversified data augmentation methods.
