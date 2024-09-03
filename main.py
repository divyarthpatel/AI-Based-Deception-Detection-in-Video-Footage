# Import necessary libraries
import cv2
import dlib
import numpy as np
from imutils import face_utils
from tensorflow.keras.models import load_model
import librosa
import numpy as np

# Load pre-trained emotion detection model
# Ensure you have a suitable model; here, we use a placeholder model path
emotion_model = load_model('emotion_model.h5')

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to detect facial emotions
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Extract Region of Interest (ROI) for emotion detection
        (x, y, w, h) = cv2.boundingRect(np.array([shape[36:48]]))
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        
        # Predict emotion using the pre-trained model
        emotion_prediction = emotion_model.predict(roi)
        emotion_label = np.argmax(emotion_prediction)
        emotion_text = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'][emotion_label]
        print(f'Emotion Detected: {emotion_text}')
        return emotion_text

# Function to analyze audio tone (basic)
def analyze_audio_tone(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path)
    
    # Extract audio features like pitch and energy
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])  # Simplistic pitch extraction
    
    # Example basic pitch threshold; normally, you'd compare against trained thresholds
    if pitch > 200:  # Arbitrary threshold for detecting a high pitch, indicating stress
        tone_analysis = "High Pitch Detected - Possible Stress"
    else:
        tone_analysis = "Normal Pitch"
    
    print(f'Audio Analysis Result: {tone_analysis}')
    return tone_analysis

# Main function to run video and audio analysis
def analyze_deception(video_path, audio_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze the current frame for emotion
        emotion_text = detect_emotion(frame)
        
        # Display the result
        cv2.putText(frame, f'Emotion: {emotion_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Deception Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Analyze the associated audio tone
    audio_result = analyze_audio_tone(audio_path)
    print(f'Final Analysis: {emotion_text}, {audio_result}')

# Example usage
# Replace with your video and audio file paths
analyze_deception('sample_video.mp4', 'sample_audio.wav')
5
