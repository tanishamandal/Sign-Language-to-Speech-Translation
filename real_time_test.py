import cv2
import numpy as np
import tensorflow as tf
import os

def real_time_test():
    # Load the trained model
    model_path = 'sign_language_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found! Please run train_model.py first.")
        return

    model = tf.keras.models.load_model(model_path)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            if height < 400 or width < 400:
                print("Error: Camera resolution too low (minimum 400x400 required)")
                break
                
            # Extract ROI with safety checks
            roi = gray[100:400, 100:400]
            if roi.size == 0:
                print("Error: Invalid region of interest")
                break
                
            # Process and predict
            roi = cv2.resize(roi, (28,28))
            input_img = roi.reshape(1,28,28,1).astype('float32') / 255.0
            prediction = model.predict(input_img)
            
            char_pred = chr(65 + np.argmax(prediction))
            confidence = np.max(prediction)
            
            # Display results
            cv2.putText(frame, f"{char_pred} ({confidence:.0%})", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow('Live Test - Press Q to quit', frame)
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_test()
