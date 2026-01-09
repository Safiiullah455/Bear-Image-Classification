import numpy as np
import pandas as pd
from keras.preprocessing import image
from PIL import ImageFile
import os
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Reshape
# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Function to load and resize image
def load_and_resize_image(path, target_size=(64, 64)):
    try:
        img = image.load_img(path, target_size=target_size, color_mode='rgb')
        return img
    except (OSError, ValueError) as e:
        print(f"Skipping corrupted or truncated image: {path}")
        return None  # Return None for corrupted images


# Function to load data from CSV
def load_data_from_csv(csv_file):
    # Read CSV file (skip header)
    data = pd.read_csv(csv_file)  # pandas will automatically handle the header row
    image_paths = data['file_path'].values  # Column name 'file_path'
    labels = data['label'].values  # Column name 'label'

    images = []
    for path in image_paths:
        # Construct full path
        full_path = os.path.join('/Bear_project', path)
        img = load_and_resize_image(full_path)
        if img is not None:
            # Convert to array and normalize
            img = image.img_to_array(img)
            img = img / 255.0
            images.append(img)

    # Convert list of images to numpy array
    images = np.array(images)

    # Ensure labels are numeric
    labels = np.nan_to_num(labels, nan=0).astype(int)  # Replace NaN with 0 and convert to int

    # Ensure labels are one-hot encoded (2 classes assumed)
    labels = to_categorical(labels, num_classes=2)

    return images, labels


# Load data from CSV
csv_file = 'Beardata.csv'  # Update with your actual CSV path
images, labels = load_data_from_csv(csv_file)

# After loading images and labels, check their lengths
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(labels)}")

# Proceed if lengths match
if len(images) == len(labels):
    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
else:
    print("Mismatch between the number of images and labels!")

# Build CNN model
cnn = Sequential()
cnn.add(Conv2D(64, (5, 5), activation='relu', input_shape=(64, 64, 3)))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(128, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))  # Hidden layer with 64 units
cnn.add(Dense(2, activation='softmax'))  # Output layer with 2 units for binary classification

# Compile the model
cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
cnn.fit(X_train, Y_train, batch_size=50, epochs=20, verbose=1)

# Evaluate the model on test data
loss, accuracy = cnn.evaluate(X_test, Y_test)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
# Save the trained model
cnn.save('BearCnn.h5')
print("Model saved as 'BearCnn.h5'")