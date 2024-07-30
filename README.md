# Credit Card Fraud Detection

This is an Application which can predict if the credit card has been compramised.

## Demo

You can try out the app [here](https://huggingface.co/spaces/Thanusha/Creditcard_fraud_detection) on Hugging Face Spaces.

## Installation

To run this app locally, you need to install neccessary dependencies. Create a virtual environments and use 'pip' to install the packages listed in 'requiremenst.txt':

### Create a virtual environment
conda create -p mynev python=3.12 -y

### Actiavate the virtual environment
conda activate myenv/.

### Install the dependencies
pip install -r requirements.txt

# features
- Predicting Fraud or non-Fraud
- Three models are given to predict
- The graphs are displayed

# Built With
- Python
- Docker
- Streamlit
- MLFlow

## About the Dataset
The Dataset was downloaded from kaggle.com. These features are tranformed into new cordinates, known as principal components.
Highly Imabalnced dataset, SMOTE Analysits used to resample the data.
Both categorical and numerical types of data are present. Label encoder method is used to train -test split data.

## About the Model
supervised learning methods were used as the labels for Fraud and non Fradulent provided.
Linear Regression, Gradient Boost Algorithm and Random forest Algorithms are compared and Gradient Boost Algorithm produced the accuracy of 95%




