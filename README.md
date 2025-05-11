# AI Bicep Curl Counter

A Python application that uses your webcam and MediaPipe Pose estimation to count bicep curls in real-time. It provides visual feedback on the current stage of the curl (up/down) and the total number of repetitions.


## Features

-   **Real-time Pose Estimation:** Utilizes Google's MediaPipe Pose to detect 33 body landmarks.
-   **Bicep Curl Counting:** Tracks the angle of the elbow joints to determine curl stage and count repetitions.
-   **Visual Feedback:** Displays the current rep count, curl stage ("UP", "DOWN", "START"), and basic form feedback on the video feed.
-   **Webcam Input:** Uses your default webcam for live video processing.
-   **Cross-Platform:** Built with Python, OpenCV, and MediaPipe, making it runnable on most systems.

## Technologies Used

-   **Python 3.x**
-   **OpenCV (`opencv-python`)**: For video capture, image manipulation, and display.
-   **MediaPipe (`mediapipe`)**: For high-fidelity body pose tracking.
-   **NumPy (`numpy`)**: For numerical operations, especially angle calculations.

## Prerequisites

Before running the application, ensure you have Python 3 installed on your system. You will also need a webcam.

## Installation

1.  **Clone the repository (or download the `bicep_curl_counter.py` file):**
    ```bash
    git clone <repository_url> # If you create a git repo
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install numpy opencv-python mediapipe
    ```

## Usage

1.  Navigate to the directory where `bicep_curl_counter.py` is located.
2.  Ensure your virtual environment is activated (if you created one).
3.  Run the script from your terminal:
    ```bash
    python bicep_curl_counter.py
    ```
4.  A window will open showing your webcam feed with pose landmarks and the curl counter.
5.  Perform bicep curls facing the camera. Ensure your shoulders, elbows, and wrists are visible.
6.  Press the **'q'** key to quit the application.

## How It Works

1.  **Video Capture:** OpenCV captures video frames from the webcam.
2.  **Image Preprocessing:** Each frame is flipped horizontally (for a mirror view) and converted from BGR (OpenCV's default) to RGB (MediaPipe's required format).
3.  **Pose Estimation:** MediaPipe's Pose model processes the RGB image to detect 33 human body landmarks.
4.  **Landmark Extraction:** Coordinates for the shoulders, elbows, and wrists (for both left and right arms) are extracted.
5.  **Angle Calculation:** A custom function `calculate_angle` determines the angle at each elbow joint using the shoulder, elbow, and wrist coordinates.
6.  **Curl Logic:**
    -   The application defines two stages: "down" (arms extended, elbow angle > 160 degrees) and "up" (arms flexed, elbow angle < 30 degrees).
    -   A repetition is counted when the user transitions from the "down" stage to the "up" stage for both arms simultaneously.
7.  **Visualization:**
    -   The detected pose landmarks and connections are drawn on the frame.
    -   A status box displays the current repetition count, the curl stage, and basic form feedback.
    -   The processed frame is displayed in an OpenCV window.

## Customization and Potential Improvements

-   **Single Arm Tracking:** Modify the logic to track curls for each arm independently or focus on a dominant arm.
-   **Advanced Form Correction:** Implement more sophisticated feedback based on other joint angles (e.g., shoulder movement, back posture).
-   **Adjustable Thresholds:** Allow users to configure the angle thresholds for "up" and "down" stages.
-   **Sound Feedback:** Add audio cues for rep counts or form corrections.
-   **Save Workout Data:** Log workout sessions (reps, duration) to a file.
-   **GUI:** Develop a more user-friendly graphical interface instead of the basic OpenCV window.
-   **Different Exercises:** Adapt the angle calculation and logic to track other exercises (e.g., squats, shoulder presses).
