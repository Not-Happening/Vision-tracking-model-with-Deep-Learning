# Pupil Movement Tracking with Deep Learning

## Overview
This project is focused on building a deep learning model to track the movement of the pupil in video frames. It uses a convolutional neural network (CNN) to predict the coordinates of the pupil in each frame of a video.

## Technologies Used
- **Python**: The programming language used to develop the entire project.
- **OpenCV**: Used for video processing, frame extraction, and image preprocessing tasks like converting images to grayscale and resizing.
- **Pandas**: Used for handling and processing the annotation data (pupil coordinates).
- **NumPy**: Utilized for efficient array operations and data manipulation.
- **TensorFlow & Keras**: The deep learning framework used to create, train, and evaluate the convolutional neural network (CNN) model.
- **Pickle**: Used to save and load the trained model for later use.
- **Scikit-Learn**: Used for splitting the data into training and validation sets.

## Model Architecture
- **Convolutional Neural Network (CNN)**:
  - **Input Layer**: Processes images of size 64x64 pixels with 1 channel (grayscale).
  - **Conv2D Layers**: Extracts features from the images using convolution operations.
  - **MaxPooling2D Layers**: Reduces the spatial dimensions of the feature maps.
  - **Flatten Layer**: Flattens the 2D matrices into a 1D vector.
  - **Dense Layers**: Fully connected layers that output the final prediction.
  - **Output Layer**: A Dense layer with 2 units representing the predicted x and y coordinates of the pupil.

## Project Structure
- **videos/**: Directory containing the input video files.
- **cords/**: Directory containing the text files with corresponding pupil coordinates.
- **pupil_model_3.h5**: The saved trained model in HDF5 format.
- **pupil_model_.pkl**: The saved trained model in Pickle format.

## DataSet
- ** Download the LPW dataset or any other data set with videos of pupil moments along with the coordinates of the pupil center in each frame.

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Not-Happening/Vision-tracking-model-with-Deep-Learning.git

## Team
1. Sreejitha Chidipothu @Not-Happening
2. Sanjana Lankadi @User-546

## Contributions:
- ** Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or suggestions.

