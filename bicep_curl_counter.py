# bicep_curl_counter.py

import numpy as np
import cv2  # OpenCV for computer vision tasks
import mediapipe as mp # Google's MediaPipe for pose estimation

# --- Helper Function to Calculate Angle ---
def calculate_angle(a, b, c):
    """
    Calculates the angle between three 2D points (forming a joint).
    Args:
        a: Numpy array or list representing the first point [x, y].
        b: Numpy array or list representing the middle point (vertex) [x, y].
        c: Numpy array or list representing the third point [x, y].
    Returns:
        angle: The calculated angle in degrees.
    """
    a = np.array(a)  # Convert to numpy array (e.g., shoulder)
    b = np.array(b)  # Convert to numpy array (e.g., elbow - vertex)
    c = np.array(c)  # Convert to numpy array (e.g., wrist)
    
    # Calculate vectors from b to a and b to c
    # vector_ba = a - b
    # vector_bc = c - b
    # angle_rad = np.arccos(np.dot(vector_ba, vector_bc) / (np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)))
    # angle_deg = np.degrees(angle_rad)
    # return angle_deg
    
    # Using atan2 for more robust angle calculation (handles full 0-360 range and quadrant issues)
    # The angle is calculated between the vector b-a and b-c
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi) # Convert to degrees and take absolute value
    
    # Ensure angle is within the 0-180 degree range for typical joint angles
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# --- Main Bicep Curl Tracking Function ---
def bicep_curl_tracker():
    """
    Main function to run the bicep curl tracker using webcam feed and MediaPipe Pose.
    """
    # Initialize MediaPipe drawing utility and Pose model
    mp_drawing = mp.solutions.drawing_utils  # Utility to draw landmarks and connections
    mp_pose = mp.solutions.pose             # MediaPipe Pose model

    # Initialize webcam
    # 0 is usually the default built-in webcam. Change if you have multiple cameras.
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return # Exit if camera can't be opened

    # Curl counter variables
    counter = 0          # To store the number of completed bicep curls
    stage = None         # To store the current stage of the curl ('up' or 'down')
    feedback = ""        # To provide feedback to the user

    # Define drawing specifications for landmarks and connections
    # Customize color (BGR), thickness, and circle radius for landmarks
    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2)
    # Customize color (BGR) and thickness for connections
    connection_drawing_spec = mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)

    # Setup MediaPipe Pose instance
    # min_detection_confidence: Minimum confidence value (0.0-1.0) for pose detection to be considered successful.
    # min_tracking_confidence: Minimum confidence value (0.0-1.0) for pose landmarks to be considered tracked successfully.
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_estimator:
        try:
            # Main loop to process video frames
            while cap.isOpened():
                # Read a frame from the webcam
                # ret: boolean, True if frame is read correctly
                # frame: the actual image array
                ret, frame = cap.read()
                if not ret:
                    print("Error: Can't receive frame (stream end?). Exiting ...")
                    break # Exit if no frame is received

                # Flip the frame horizontally for a more natural selfie-view (mirror effect)
                frame_flipped = cv2.flip(frame, 1)
                
                # --- MediaPipe Pose Processing ---
                # Convert the BGR image (OpenCV default) to RGB (MediaPipe requirement)
                image_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
                
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image_rgb.flags.writeable = False
                
                # Process the RGB image with MediaPipe Pose to detect landmarks
                results = pose_estimator.process(image_rgb)
                
                # Note: We will draw on `frame_flipped` (BGR) later, so no need to convert `image_rgb` back yet.

                # --- Landmark Extraction and Angle Calculation ---
                try:
                    # Check if any pose landmarks were detected
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        
                        # Get coordinates for LEFT arm joints
                        # landmarks[mp_pose.PoseLandmark.LANDMARK_NAME.value] gives the landmark object
                        # .x, .y, .z are normalized coordinates (0.0 to 1.0)
                        # .visibility is the likelihood of the landmark being visible
                        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        # Get coordinates for RIGHT arm joints
                        shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        
                        # Calculate the angle of the left and right elbows
                        angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                        angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
                        
                        # --- Bicep Curl Counting Logic ---
                        # This logic assumes both arms are performing the curl.
                        # You might want to modify this to track arms independently or focus on one.
                        
                        # Check if both arms are in the "down" position (extended)
                        if angle_l > 160 and angle_r > 160:
                            stage = "down"
                            feedback = "Lower further" if (angle_l < 170 or angle_r < 170) else "Arms Down"
                        
                        # Check if both arms are in the "up" position (flexed) and were previously "down"
                        if angle_l < 30 and angle_r < 30 and stage =='down':
                            stage = "up"
                            counter += 1
                            print(f"Reps: {counter}") # Print to console
                            feedback = "Curl Up!"
                        elif angle_l < 60 and angle_r < 60 and stage == 'down': # Partial up
                            feedback = "Curl higher!"
                        
                        # Provide feedback on angle if not fully up or down
                        if stage == "up" and (angle_l > 40 or angle_r > 40):
                            feedback = "Lower slowly"

                    else:
                        # If no landmarks are detected
                        feedback = "No pose detected"
                        # Optionally, reset stage if no landmarks are visible for a while
                        # stage = None 

                except Exception as e:
                    # print(f"Error processing landmarks: {e}") # Uncomment for debugging
                    feedback = "Error in detection"
                    pass # Continue to next frame if error in landmark processing
                
                # --- Drawing UI Elements on the Frame ---
                # Draw a status box for displaying reps and stage
                # Rectangle: cv2.rectangle(image, start_point, end_point, color_BGR, thickness)
                # -1 thickness fills the rectangle
                cv2.rectangle(frame_flipped, (0,0), (300,100), (245,117,16), -1) # Adjusted width for feedback
                
                # Display "REPS" text
                cv2.putText(frame_flipped, 'REPS', (15,25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                # Display the current rep count
                cv2.putText(frame_flipped, str(counter), 
                            (20,80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
                # Display "STAGE" text
                cv2.putText(frame_flipped, 'STAGE', (120,25), # Adjusted X position
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
                # Display the current stage (or "START" if no stage yet)
                cv2.putText(frame_flipped, stage if stage else "START", 
                            (100,80), # Adjusted X position
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5 if stage else 1, (255,255,255), 2, cv2.LINE_AA) # Adjusted size

                # Display feedback
                cv2.putText(frame_flipped, feedback, (10, 130), # Position for feedback
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


                # Draw pose landmarks on the frame_flipped (BGR image)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame_flipped, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            landmark_drawing_spec, connection_drawing_spec)               
                
                # Display the processed frame
                cv2.imshow('Mediapipe Bicep Curl Tracker', frame_flipped)

                # Wait for a key press (10 milliseconds)
                # 0xFF is a bitmask to get the last 8 bits, ensuring cross-platform compatibility for ord()
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Exiting program...")
                    break # Exit loop if 'q' is pressed
        finally:
            # Release resources
            print("Releasing resources...")
            if cap.isOpened():
                cap.release()         # Release the webcam
            cv2.destroyAllWindows() # Close all OpenCV windows
            print("Resources released.")

# --- Script Execution ---
if __name__ == '__main__':
    # This block runs when the script is executed directly
    bicep_curl_tracker()
