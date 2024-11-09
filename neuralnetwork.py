import numpy as np
import pandas as pd

'''
# Initialize a vector of 4 rows for the hidden layer, each row (element) corresponds to one hidden neuron. And also the vector B.
# Initialize a 3x4 parameters matrix (3: input features, 4: no. of neurons in the hidden layer).

We want to do a for loop (for now) on the dataset to update this vector.

Z = W.T.dot(X) + B
    --> X denotes to one training example [x1, x2, x3]
    --> W denotes to a 3x4 parameters matrix (3: input features, 4: no. of neurons in the hidden layer).
    --> B denotes to the bias vector (4x1, each row corresponds to one neuron in the hidden layer).

And, we will keep updating it based on the training data using a for loop (for now).
'''

Z = np.zeros((4, 1))
W = np.random.randn(3, 4)
B = np.zeros((4, 1))

# Load the dataset
df = pd.read_csv('data/house_prices_dataset.csv')

for index, row in df.iterrows():
    x1 = row['Bedrooms']
    x2 = row['Area']
    x3 = row['Neighborhood_Wealth']

    X = np.array([x1, x2, x3])
    Z = W.T.dot(X) + B

    price = row['Price']
    #print(x2, " ", price)
