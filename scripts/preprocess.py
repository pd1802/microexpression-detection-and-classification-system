import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"D:\gitcode\emotion\microexpression-detection-and-classification-system\data\fer2013.csv")

# Ensure directories exist
plot_dir = "saved_plots"
data_dir = "data"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Check dataset distribution
sns.countplot(x=df["emotion"])
plt.title("Emotion Distribution")
plt.savefig(os.path.join(plot_dir, "emotion_distribution.png"))
plt.show()

# Emotion labels
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Convert pixel strings to numpy arrays
def preprocess_images(pixels):
    img_array = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
    img_array = img_array / 255.0  # Normalize pixel values (0-1)
    return img_array

df["pixels"] = df["pixels"].apply(preprocess_images)

# Convert to numpy arrays
X = np.stack(df["pixels"].values)  # Convert list to numpy array
y = to_categorical(df["emotion"].values, num_classes=7)  # One-hot encode labels

# Reshape for CNN
X = X.reshape(-1, 48, 48, 1)  # (samples, height, width, channels)

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
np.save(os.path.join(data_dir, "X_train.npy"), X_train)
np.save(os.path.join(data_dir, "X_test.npy"), X_test)
np.save(os.path.join(data_dir, "y_train.npy"), y_train)
np.save(os.path.join(data_dir, "y_test.npy"), y_test)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
