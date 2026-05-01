# ♟️ AI Automated Chess Training System  
Computer Vision Driven Robotic Arm for Real-Time Chess Automation  

## 📌 Overview  
This project presents an AI-powered chess training system that integrates **computer vision**, **robotics**, and **AI decision-making**.  
It detects the chessboard state using a YOLO model and executes moves physically using a gantry robotic arm.

---

## 🚀 Features  
- Real-time chessboard and piece detection (YOLO)  
- FEN generation from visual input  
- AI-based move decision (chess engine logic)  
- Robotic arm execution (pick & place using gripper)  
- Redis-based state management  
- Modular and scalable architecture  

---

## 🧠 System Architecture  

### Workflow:
1. `camera_capture.py` → Captures frames from camera  
2. `aruco_calibration.py` → Calibrates board using ArUco markers  
3. `yolo_fen.py` → Detects pieces & generates FEN  
4. `chess_brain.py` → Computes best move  
5. `move_validator.py` → Validates legality  
6. `main.py` → Controls full pipeline & robot execution  

---

## ⚙️ Tech Stack  

### Software:
- Python  
- YOLO (Ultralytics)  
- OpenCV  
- Redis  

### Hardware:
- Arduino Controller  
- Stepper Motors + Drivers  
- Gantry Robotic Arm  
- Gripper Module  

