import pandas as pd
import numpy as np

# Number of samples in the dataset
num_samples = 100

# Generating random data
np.random.seed(0)
data = {
    "Keyword Frequency": np.random.rand(num_samples) * 0.2,
    "Word Count": np.random.randint(50, 300, num_samples),
    "Exclamation Frequency": np.random.rand(num_samples) * 0.15,
    "Spam Domain": np.random.randint(0, 2, num_samples),
    "Uppercase Ratio": np.random.rand(num_samples) * 0.3,
    "Label": np.random.randint(0, 2, num_samples)
}

# Creating a DataFrame and saving it as a CSV file
df = pd.DataFrame(data)
df.to_csv("spam_dataset.csv", index=False)
