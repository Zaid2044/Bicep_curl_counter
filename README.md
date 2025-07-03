# Bicep Curl Counter 💪 (Pose Estimation)

**Author:** MOHAMMED ZAID AHMED  
**Repository:** [Bicep\_curl\_counter](https://github.com/Zaid2044/Bicep_curl_counter)

A computer vision-powered application that uses pose estimation to count bicep curls in real time using your webcam. Built with MediaPipe and OpenCV, it provides an interactive and accurate workout tracking experience.

---

## 🔍 Features

* 🎯 Real-time bicep curl detection using elbow angle
* 📸 Live webcam tracking via OpenCV
* 📈 Repetition counter with stage detection ("Up", "Down")
* 🏋️‍♂️ Instant feedback on form and reps
* 🧠 Lightweight and runs on CPU

---

## 🧠 Tech Stack

* **Language:** Python
* **Libraries:** OpenCV, MediaPipe, NumPy
* **Platform:** Desktop (Webcam required)

---

## 🛠️ How It Works

* Detects key landmarks using **MediaPipe Pose**
* Calculates **elbow angle** from shoulder, elbow, and wrist positions
* Tracks motion range to count **reps** and display **status**
* Visual feedback rendered on webcam feed

---

## 🚀 Setup Instructions

1. **Clone the repo**

   ```bash
   git clone https://github.com/Zaid2044/Bicep_curl_counter.git
   cd Bicep_curl_counter
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   python main.py
   ```

---

## 📊 Demo

<p align="center">
  <img src="docs/demo.gif" alt="Bicep Curl Counter Demo" width="700"/>
</p>

---

## 🔮 Future Additions

* Add squats, pushups, and shoulder press counters
* Voice feedback with rep count
* Save workout logs to CSV
* Optional GUI overlay with Tkinter or Flask

---

## 📜 License

This project is licensed under the MIT License.

---
