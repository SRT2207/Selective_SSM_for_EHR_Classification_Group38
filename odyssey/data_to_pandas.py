#!/usr/bin/python3
import os
import json
import pandas as pd
import numpy as np
import tarfile

base_path = './'

# Define the path to the P12data directory
data_path = 'P12data/'

# List all .json files in the directory
json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]

# Load each JSON file into a DataFrame with the same name as the file
json_df = {}
for file in json_files:
    file_path = os.path.join(data_path, file)
    with open(file_path, 'r') as f:
        data = json.load(f)
        # Normalize JSON data into a DataFrame
        df = pd.json_normalize(data)
        # Store in a dictionary with the key as the filename
        json_df[file.replace('.json', '')] = df

# Display the keys of the dictionary
print("DataFrames created:", list(json_df.keys()))


# Dictionary to store DataFrames with the name of the .npy file as the key
dataframes = {}

# List all .tar.gz files in the directory
tar_files = [f for f in os.listdir(data_path) if f.endswith('.tar.gz')]

# Loop through each .tar.gz file, extract, and load .npy data into DataFrames
for tar_file in tar_files:
    tar_path = os.path.join(data_path, tar_file)
    extract_folder = os.path.join(data_path, tar_file.replace('.tar.gz', ''))

    # Create directory for extracted files
    os.makedirs(extract_folder, exist_ok=True)

    # Extract the tar.gz file
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_folder)

    # Load each .npy file in the extracted folder
    for root, _, files in os.walk(extract_folder):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if the file is a .npy file
            if file.endswith('.npy'):
                # Load the .npy file with allow_pickle=True
                array_data = np.load(file_path, allow_pickle=True)

                # Convert to DataFrame
                df = pd.DataFrame(array_data)

                # Store the DataFrame in the dictionary using the .npy file's name (without extension) as the key
                file_name_without_extension = file.replace('.npy', '')
                dataframes[file_name_without_extension] = df

train_dfs = list()
test_dfs = list()
validation_dfs = list()

for name in list(dataframes.keys()):
    if name.startswith('train'):
        train_dfs.append(dataframes[name])
    elif name.startswith('test'):
        test_dfs.append(dataframes[name])
    elif name.startswith('validation'):
        validation_dfs.append(dataframes[name])

pd.concat(train_dfs).to_pickle(base_path+'odyssey/P12data/train_df.pkl')
pd.concat(test_dfs).to_pickle(base_path+'odyssey/P12data/test_df.pkl')
pd.concat(validation_dfs).to_pickle(base_path+'odyssey/P12data/validation_df.pkl')

# Display the keys of the dictionary to verify the DataFrames are named correctly
print("DataFrames created:", list(dataframes.keys()))