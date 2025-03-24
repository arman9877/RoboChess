# ♟️ RoboChess

RoboChess is a robotic chess-playing system that integrates:
- **Kinova Gen3 Lite robotic arm** (using Kortex API)
- **Computer vision** (OpenCV)
- **Stockfish chess engine**
- **ROS (Robot Operating System)** on Ubuntu

The robot observes the chessboard, decides on the best move using Stockfish, and physically executes the move using the Kinova arm. It also supports playing against a human.

---

## 🚀 Features

- Autonomous chess move detection and execution
- Play against Stockfish or a human player
- Image processing for board and piece detection
- Physical robot control via high-level Kortex API
- Real-time board state updates

---

## 🧩 Components

| File | Description |
|------|-------------|
| `Selfplay.py` | Robot plays chess against Stockfish autonomously |
| `Play_with_Human.py` | Human vs Robot with camera input and GUI buttons |
| `Board_setup_XY(1H).py` | Moves robot through a sequence of calibration positions |
| `Board_setup_with_Camera.py` | Perspective transformation from camera to chessboard view |
| `Arm_Home.py` | Sends robot arm to the home position and runs kinematic checks |
| `Camera_indice.py` | Detects available camera indices |
| `chessboard_positions_NC.csv` | Maps chess squares to joint angles for movement |

---

## 🛠️ Requirements

- Ubuntu 20.04 or later
- Python 3.7+
- ROS (Kinetic/Melodic recommended for Kinova)
- Kinova Kortex API (Python)
- [Stockfish](https://stockfishchess.org/)
- OpenCV, pandas, numpy, pyautogui

Install dependencies:
```bash
pip install opencv-python pandas numpy pyautogui stockfish

