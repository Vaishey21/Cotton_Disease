import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from shutil import copyfile

# Define constants
IMAGE_SIZE = (224, 224)  # Resize images to 224x224 pixels
DATA_DIRECTORY = r"D:\New folder\leaf_disease"
TRAIN_DIRECTORY = r"D:\New folder\train"
TEST_DIRECTORY = r"D:\New folder\test"
VALID_DIRECTORY = r"D:\New folder\validate"

# Function to load and preprocess images
def load_and_preprocess_data(data_directory):
    images = []
    labels = []
    
    # Loop through each subdirectory (class)
    for subdir in os.listdir(data_directory):
        class_path = os.path.join(data_directory, subdir)
        if os.path.isdir(class_path):
            # Loop through images in each subdirectory
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                if os.path.isfile(image_path):
                    # Read and resize image
                    try:
                        image = cv2.imread(image_path)
                        # Resize image
                        if image is not None:
                            image = cv2.resize(image, IMAGE_SIZE)
                        else:
                            print(f"Could not read image: {image_path}")
                            continue
                    except Exception as e:
                        print(f"Error reading image: {image_path}")
                        print(e)
                        continue
                    
                    # Normalize pixel values (0-1)
                    image = image.astype('float32') / 255.0
                    
                    # Append image and corresponding label
                    images.append(image)
                    labels.append(subdir)
    
    return np.array(images), np.array(labels)

# Load and preprocess data
images, labels = load_and_preprocess_data(DATA_DIRECTORY)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Split the training data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create directories to save data
os.makedirs(TRAIN_DIRECTORY, exist_ok=True)
os.makedirs(TEST_DIRECTORY, exist_ok=True)
os.makedirs(VALID_DIRECTORY, exist_ok=True)

# Function to save images to directory
def save_images_to_directory(images, labels, directory):
    for i, (image, label) in enumerate(zip(images, labels)):
        class_directory = os.path.join(directory, str(label))
        os.makedirs(class_directory, exist_ok=True)
        image_path = os.path.join(class_directory, f"{i}.jpg")
        cv2.imwrite(image_path, cv2.cvtColor(image * 255, cv2.COLOR_RGB2BGR))

# Save training, testing, and validation images to directories
save_images_to_directory(X_train, y_train, TRAIN_DIRECTORY)
save_images_to_directory(X_test, y_test, TEST_DIRECTORY)
save_images_to_directory(X_valid, y_valid, VALID_DIRECTORY)

print("Data split and saved successfully.")
