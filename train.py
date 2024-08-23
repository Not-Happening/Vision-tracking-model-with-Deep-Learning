import cv2
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import pickle

img_ht, img_wd = 64, 64
batch = 6  
video = 'videos/'
coord = 'cords/'

def create():
    Model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_ht, img_wd, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2)
    ])
    Model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return Model
def extracting(vid_path):
    capture = cv2.VideoCapture(vid_path)
    Frames = []
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        Frames.append(frame)
    capture.release()
    return Frames

def load_annots(annot_path):
    annotations = pd.read_csv(annot_path, sep=' ', header=None, names=['x', 'y'])
    return annotations

def preprocess(Frames):
    preprocessed_frames = []
    for frame in Frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_wd, img_ht))
        normalized = resized / 255.0
        preprocessed_frames.append(normalized)
    return np.array(preprocessed_frames)

def save_model(Model, TATA):
    Model.save(TATA)
def load_model(TATA):
    return tf.keras.models.load_model(TATA)


def save_model(model, TATA):
    with open(TATA, 'wb') as f:
        pickle.dump(Model, f)
def load_model(TATA):
    with open(TATA, 'rb') as f:
        return pickle.load(f)

video_files = sorted([os.path.join(video, f) for f in os.listdir(video) if f.endswith('.avi')])
coord_files = sorted([os.path.join(coord, f) for f in os.listdir(coord) if f.endswith('.txt')])
model_filename = 'pupil_model_3.h5'
if os.path.exists(model_filename):
    Model = load_model(model_filename)
else:
    Model = create()

for i in range(0, len(video_files), batch):
    batch_videos = video_files[i:i + batch]
    batch_coords = coord_files[i:i + batch]
    X_batch = []
    y_batch = []
    for vid_path, annot_path in zip(batch_videos, batch_coords):
        Frames = extracting(vid_path)
        preprocessed_frames = preprocess(Frames)
        annotations = load_annots(annot_path)
        annotations['x'] = annotations['x'].apply(pd.to_numeric, errors='coerce')
        annotations['y'] = annotations['y'].apply(pd.to_numeric, errors='coerce')
        labels = annotations[['x', 'y']].values
        X_batch.extend(preprocessed_frames)
        y_batch.extend(labels)
    X_batch = np.array(X_batch).reshape(-1, img_ht, img_wd, 1)
    y_batch = np.array(y_batch)
    X_train, X_val, y_train, y_val = train_test_split(X_batch, y_batch, test_size=0.2, random_state=42)
    Model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
    save_model(Model, model_filename)

pickle_filename = 'pupil_model_.pkl'
save_model(Model, pickle_filename)

Model = load_model(pickle_filename)

'''# Example usage with a new frame for prediction
def predict_gaze_direction(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_width, img_height))
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=(0, -1))
    prediction = model.predict(input_data)
    return prediction[0]

# Predict pupil coordinates for each frame in the test video
test_video_path = '16.avi'
test_frames = extracting_frames(test_video_path)

for frame in test_frames:
    gaze_direction = predict_gaze_direction(frame, model)
    print(f"Gaze direction (x, y): {gaze_direction}")
    
    # Optional: Draw the predicted pupil position on the frame
    x, y = int(gaze_direction[0]), int(gaze_direction[1])
    cv2.circle(frame, (x, y), 25, (0, 255, 0), -1)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()'''
