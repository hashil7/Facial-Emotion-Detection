import numpy as np
import cv2
import tensorflow as tf

# Load the pretrained model
model = tf.keras.models.load_model('new_archi_model.h')

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Access webcam stream
cap = cv2.VideoCapture(0)  # 0 for the default webcam, or specify the index of the desired webcam

# Dictionary for mapping label indices to emotion names
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face region from the frame
        face_roi = frame_gray[y:y+h, x:x+w]
        
        # Resize face region to match model input size
        face_resized = cv2.resize(face_roi, (48, 48))
        
        # Normalize pixel values to [0, 1]
        face_normalized = face_resized / 255.0
        
        # Reshape the face to match the expected input shape of your model
        input_face = np.expand_dims(face_normalized, axis=-1)
        input_face = np.expand_dims(input_face, axis=0)
        
        # Make prediction
        prediction = model.predict(input_face)
        
        # Extract the predicted emotion
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        
        # Display predicted emotion with confidence overlay
        cv2.putText(frame, f'{predicted_emotion} ({np.max(prediction)*100:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        # Draw a transparent overlay showing the confidence level
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y-25), (x + w, y - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam resources
cap.release()
cv2.destroyAllWindows()
