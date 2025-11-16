# Student: [Your Name/ID]
# Course: [Your Course Name]
# Date: [Current Date]

"""
prepare_data.py

This script is a utility to split our main food data into training and testing sets.
This is an important step in machine learning to ensure our algorithms generalize
well to new, unseen data. We run this once to create the data files needed
by our optimization algorithms.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(file_path='foods.csv', train_path='training_data.csv', test_path='testing_data.csv', test_size=0.3, random_state=42):
    """
    Splits the food data into training and testing sets.
    It reads the main `foods.csv`, randomly splits the foods (columns),
    and saves them into two new CSV files.
    """
    try:
        # Load the dataset using pandas. The first column (nutrients) is our index.
        df = pd.read_csv(file_path, index_col=0)
        
        # Get a list of all the food names from the columns.
        food_columns = df.columns.tolist()
        
        # Use scikit-learn's train_test_split to randomly divide the foods.
        # `random_state=42` ensures we get the same split every time we run this.
        train_foods, test_foods = train_test_split(food_columns, test_size=test_size, random_state=random_state)
        
        # Create new dataframes for the training and testing sets.
        train_df = df[train_foods]
        test_df = df[test_foods]
        
        # Save the split data to new CSV files.
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        
        print(f"Data successfully split into '{train_path}' and '{test_path}'")
        print(f"Training set has {len(train_foods)} foods.")
        print(f"Testing set has {len(test_foods)} foods.")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# This part makes the script runnable from the command line.
if __name__ == "__main__":
    # When we run `python prepare_data.py`, this function will be called.
    split_data()