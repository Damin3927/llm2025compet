#%%
import os
import json

#%%
# print the filtered dataset size
data_path = "./results/filtered_dataset"

for split in os.listdir(data_path):
    collection = []
    for file in os.listdir(os.path.join(data_path, split)):
        with open(os.path.join(data_path, split, file), "r") as f:
            data = json.load(f)
        collection.append(len(data))
    print(split, sum(collection))
