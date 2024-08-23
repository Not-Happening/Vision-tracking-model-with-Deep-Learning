import cv2
import numpy as np
import pandas as pd
import os
import pickle

img_ht, img_wd = 64, 64
test_video_path = '66.avi'
test_annotation_path = '66.txt'
pickle_filename = 'pupil_model.pkl'
def load_model(TATA):
    with open(TATA, 'rb') as f:
        return pickle.load(f)
def extracting(vid_path):
    cap = cv2.VideoCapture(vid_path)
    Frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        Frames.append(frame)
    cap.release()
    return Frames
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_wd, img_ht))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=(0, -1))
def annotations(annot_path):
    return pd.read_csv(annot_path, sep=' ', header=None, names=['x', 'y'])

Model = load_model(pickle_filename)

def predict_gaze_direction(frame, Model):
    input_data = preprocess(frame)
    prediction = Model.predict(input_data)
    return prediction[0]

test_frames = extracting(test_video_path)
true_annotations = annotations(test_annotation_path)
true_annotations['x'] = true_annotations['x'].apply(pd.to_numeric, errors='coerce')
true_annotations['y'] = true_annotations['y'].apply(pd.to_numeric, errors='coerce')
predicted_coords = []
true_coords = true_annotations.values
#prediction
for frame in test_frames:
    gaze_direction = predict_gaze_direction(frame, Model)
    predicted_coords.append(gaze_direction)
    x, y = int(gaze_direction[0]), int(gaze_direction[1])
    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

predicted_coords = np.array(predicted_coords)
true_coords = np.array(true_coords)

valid_idx = ~np.isnan(true_coords).any(axis=1)
predicted_coords = predicted_coords[valid_idx]
true_coords = true_coords[valid_idx]
mae_x = np.mean(np.abs(predicted_coords[:, 0] - true_coords[:, 0]))
mae_y = np.mean(np.abs(predicted_coords[:, 1] - true_coords[:, 1]))

print(f"Mean Absolute Error (X): {mae_x}")
print(f"Mean Absolute Error (Y): {mae_y}")
