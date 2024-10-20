import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# Set model parameters
img_height, img_width = 180, 180

# Define class names
class_names = [
    'freshbanana', 'freshcapsicum', 'freshcucumber', 'freshoranges',
    'freshpotato', 'freshtomato', 'rottenbanana', 'rottencapsicum',
    'rottencucumber', 'rottenoranges', 'rottenpotato', 'rottentomato', 
    
]

# Map class names to produce type and freshness state
def get_fruit_and_freshness(class_name):
    if "rotten" in class_name:
        return class_name.replace("rotten", ""), "Rotten"
    elif "fresh" in class_name:
        return class_name.replace("fresh", ""), "Fresh"
    return "Unknown", "Unknown"

# Load the trained model
model = tf.keras.Sequential([
    tf.keras.applications.ResNet50(include_top=False, input_shape=(180, 180, 3), pooling='avg', weights=None),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Load model weights
model.load_weights('resnet_model_weights.weights.h5')

# Function to perform inference on a single frame
def predict_frame(frame):
    # Resize and preprocess frame
    resized_frame = cv2.resize(frame, (img_width, img_height))
    input_frame = np.expand_dims(resized_frame, axis=0)
    input_frame = preprocess_input(input_frame)

    # Predict the class probabilities
    predictions = model.predict(input_frame)[0]

    # Get detected class and confidence score
    max_index = np.argmax(predictions)
    class_name = class_names[max_index]
    confidence = predictions[max_index] * 100  # Probability percentage

    # Get the fruit/vegetable name and freshness state
    produce, freshness = get_fruit_and_freshness(class_name)

    # Calculate freshness percentage if applicable
    fresh_key = f"fresh{produce}"
    rotten_key = f"rotten{produce}"
    fresh_prob = predictions[class_names.index(fresh_key)] if fresh_key in class_names else 0
    rotten_prob = predictions[class_names.index(rotten_key)] if rotten_key in class_names else 0

    # Compute the freshness percentage
    total_prob = fresh_prob + rotten_prob
    freshness_percentage = (fresh_prob / total_prob) * 100 if total_prob > 0 else 0

    return produce, freshness, freshness_percentage, confidence

# Open the video file or camera input
input_video = cv2.VideoCapture('orange.mp4')  # Use 0 for webcam
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(input_video.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('orange_output.mp4', fourcc, fps, (frame_width, frame_height))

while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break

    # Perform prediction on the current frame
    produce, freshness, freshness_percentage, confidence = predict_frame(frame)

    # Draw bounding box, class name, and freshness percentage
    # text = f"{produce}-{freshness} & Freshness:{freshness_percentage:.2f}%"
    text = f"Freshness:{freshness_percentage:.2f}%"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Save the frame to the output video
    output_video.write(frame)

    # Display the frame (Optional, for live view)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC' key
        break

# Release resources
input_video.release()
output_video.release()
cv2.destroyAllWindows()
