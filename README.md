## DeepLearning-Basics
This Jupyter notebook provides a basic example of using deep learning for binary classification. The notebook uses TensorFlow and Keras to build an artificial neural network (ANN) to predict whether a customer will exit a bank or not based on certain features.

### Getting Started
To run this notebook, you will need to have Python installed, along with TensorFlow and other required libraries. You can install them using pip:
Copy code
```
pip install pandas matplotlib numpy tensorflow
```
Clone the repository or download the Churn_Modelling.csv file from here and place it in the same directory as this notebook.
### Usage
Open the notebook in a Jupyter environment (e.g., Google Colab) and run each cell sequentially.
The notebook will load the dataset, perform some feature engineering, split the data into training and testing sets, scale the features, build an ANN, train the model, and evaluate its performance.

### Libraries Used
pandas: For data manipulation and analysis.
matplotlib: For plotting graphs.
numpy: For numerical operations.
tensorflow: For building and training the neural network.

### Model Architecture
The neural network consists of an input layer, two hidden layers, and an output layer.
The input layer has 11 units, one for each feature.
The first hidden layer has 7 units, the second hidden layer has 6 units, both using the ReLU activation function.
The output layer has 1 unit with a sigmoid activation function, as this is a binary classification problem.

### Training
The model is trained using the Adam optimizer and binary crossentropy loss function.
Early stopping is used to prevent overfitting.
The training history is plotted to visualize the model's performance over epochs.

### Evaluation
The model is evaluated using the test set and the confusion matrix and accuracy score are calculated.

### Conclusion
This notebook provides a basic example of using deep learning for binary classification. You can further experiment with the model architecture, optimizer, and other hyperparameters to improve performance.
