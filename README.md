# deep-learning-challenge
Alphabet Soup, a nonprofit organization, needs a tool to identify the most promising funding applicants. The task is to predict whether organizations will be successful if funded by Alphabet Soup using machine learning and neural networks. A dataset with over 34,000 funded organizations is provided, containing various details like application type, affiliation, and income classification.

Step 1: Preprocess the Data

Read the charity_data.csv into a Pandas DataFrame.
Identify the target variable (what we want to predict) and the features (input variables).
Drop the identification columns (EIN and NAME).
Determine unique values for each column.
For columns with more than 10 unique values, group rare categorical variables into a new category, "Other."
Use pd.get_dummies() to encode categorical variables.
Split the data into features (X) and target (y).
Split the data into training and testing datasets using train_test_split.
Scale the features using StandardScaler.


Step 2: Compile, Train, and Evaluate the Model

Create a neural network model using TensorFlow and Keras.
Design the architecture with input features, hidden layers, and an output layer.
Choose appropriate activation functions for hidden layers and the output layer.
Compile and train the model, saving weights every five epochs.
Evaluate the model using test data to calculate loss and accuracy.
Save the results to an HDF5 file named AlphabetSoupCharity.h5.



Step 3: Optimize the Model

Experiment with optimizing the model to achieve a predictive accuracy higher than 75%.
Adjust input data, including dropping or adding columns and modifying binning strategies.
Explore adding more neurons, hidden layers, or changing activation functions.
Adjust the number of epochs during training.
Make at least three attempts at optimizing the model.
Create a new Google Colab file named AlphabetSoupCharity_Optimization.ipynb.
Import dependencies, read the dataset, and preprocess it.
Design a neural network model with modifications based on optimization attempts.
Save the optimized results to an HDF5 file named AlphabetSoupCharity_Optimization.h5.
