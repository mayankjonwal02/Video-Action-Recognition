import os
from altair import FontWeight
import cv2
from numpy import size
import streamlit as st
import tensorflow as tf
import numpy as np




def load_model():
    # Load the model using TensorFlow
    model = tf.keras.models.load_model('s16b4.h5')

    return model

model = load_model()

def main():
    st.title('Video Classification')
    video_file = st.file_uploader("choose a video....",type=["mp4","avi","mov"])
    if(video_file is not None):
       
        
        save_video(video_file)

        
   

def save_video(video_file):
    try:
        os.makedirs("uploads")
    except:
        print("already exist")
    file_path = os.path.join("uploads",video_file.name)
    with open(file_path, 'wb') as f:
        f.write(video_file.read())
    st.success("video_saved_succesfully")
    with st.container():
        col1 , col2 = st.columns(2)
        with col1:
             st.video(video_file)
        with col2:
            image_placeholder = st.empty()
            image_placeholder.write('Predicting....')
            predict(file_path, image_placeholder)


def predict(file_path,image_placeholder):
    CLASSES_LIST = ["punch","pullup","pour","pick","laugh"]
    SEQUENCE_LENGTH = 16
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(file_path)
    
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames we will extract.
    frames_list = []

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter )

        # Read a frame.
        success, frame = video_reader.read()

        # Check if frame is not read properly then break the loop.
        if not success:
            break

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (128,128))

        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255

        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    image_placeholder.write(predicted_class_name)

    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

    # Release the VideoCapture object.
    video_reader.release()
        

        
   


if __name__ == '__main__':
    main()
