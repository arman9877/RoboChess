import sys
import os
import pyautogui  # Simulates key presses
import io  # Used to capture function output
import time
import threading
import pandas as pd
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from kortex_api.Exceptions.KServerException import KServerException
import utilities

from stockfish import Stockfish
import cv2
import numpy as np

# Load positions from CSV file
CSV_FILE_PATH = "chessboard_positions_NC.csv"

# Define button properties (size, position, color)
buttons = [
    {"label": "Play", "key": 'p', "pos": (20, 50, 310, 100), "color": (255, 69, 0)},  # Vibrant Red-Orange
    {"label": "Capture", "key": 'c', "pos": (20, 120, 310, 170), "color": (255, 69, 0)},  # Vibrant Red-Orange
    {"label": "Stockfish", "key": 's', "pos": (20, 190, 155, 240), "color": (64, 200, 150)},  # Turquoise
    {"label": "Move Arm", "key": 'm', "pos": (175, 190, 310, 240), "color": (64, 200, 150)},  # Turquoise
    {"label": "Undo", "key": 'u', "pos": (20, 260, 155, 310), "color": (64, 150, 150)},  # Turquoise
    {"label": "Quit", "key": 'q', "pos": (175, 260, 310, 310), "color": (65, 105, 225)}  # Royal Blue
]


def load_positions_from_csv(file_path):
    raw_data = pd.read_csv(file_path, header=None)
    raw_data_combined = raw_data.astype(str).apply(' '.join, axis=1)
    parsed_data = raw_data_combined.str.extract(r'(?P<position>\w+):\[(?P<angles>[0-9.\s]+)\]')
    parsed_data['angles'] = parsed_data['angles'].apply(lambda x: list(map(float, x.split())) if pd.notnull(x) else None)
    parsed_data = parsed_data.dropna()
    return parsed_data.set_index('position')['angles'].to_dict()

POSITIONS_MAP = load_positions_from_csv(CSV_FILE_PATH)

ROBOT_IP = "192.168.1.10"
USERNAME = "admin"
PASSWORD = "admin"
TIMEOUT_DURATION = 20

# Set the QT platform plugin to xcb to resolve the Wayland issue
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Initialize Stockfish engine
stockfish = Stockfish(path="/usr/games/stockfish", parameters={
    "Threads": 2,
    "Skill Level": 1,
})

move_history = []  # Store all moves

def check_for_end_or_abort(e):
    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
            e.set()
    return check

def set_actuator_positions_directly(base_client, angles):
    print(f"Setting actuator angles to: {angles}")
    action = Base_pb2.Action()
    action.name = "Set Actuator Angles"
    action.application_data = ""

    for i, angle in enumerate(angles):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = i
        joint_angle.value = angle

    e = threading.Event()
    notification_handle = base_client.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base_client.ExecuteAction(action)

    print("Waiting for movement to finish...")
    finished = e.wait(TIMEOUT_DURATION)
    base_client.Unsubscribe(notification_handle)

    if finished:
        print("Actuator angles set successfully.")
    else:
        print("Timeout on action notification wait.")
    return finished

def set_gripper_position(base_client, position):
    """Set the gripper position (0.0 to 1.0)."""
    print(f"Setting gripper to position: {position*100}%")
    gripper_command = Base_pb2.GripperCommand()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger = gripper_command.gripper.finger.add()
    finger.finger_identifier = 1
    finger.value = position
    base_client.SendGripperCommand(gripper_command)
    time.sleep(1)  # Allow gripper to complete the movement

def get_movement_sequence(input_move, capture=False, castling=False, promotion=False):
    if promotion:
        start = input_move[:2]
        end = input_move[2:]
        return [
            # Remove the piece from the start square
            f"{start}2", f"{start}1", "CLOSE_GRIPPER", f"{start}2", "OUT", "OPEN_GRIPPER",
            # Pick up the promotion piece
            "PROM2", "PROM1", "CLOSE_GRIPPER", "PROM2",
            # Place the promotion piece on the end square
            f"{end}2", f"{end}1", "OPEN_GRIPPER", f"{end}2", "HOME"
        ]

    if castling:
        king_start, king_end = input_move[:2], input_move[2:]
        if king_start == "e1" and king_end == "g1":  # Kingside castling white
            rook_start, rook_end = "h1", "f1"
        elif king_start == "e1" and king_end == "c1":  # Queenside castling white
            rook_start, rook_end = "a1", "d1"
        elif king_start == "e8" and king_end == "g8":  # Kingside castling black
            rook_start, rook_end = "h8", "f8"
        elif king_start == "e8" and king_end == "c8":  # Queenside castling black
            rook_start, rook_end = "a8", "d8"
        else:
            raise ValueError("Invalid castling move")

        return [
            # Move king
            f"{king_start}2", f"{king_start}1", "CLOSE_GRIPPER", f"{king_start}2", f"{king_end}2", f"{king_end}1", "OPEN_GRIPPER", f"{king_end}2",
            # Move rook
            f"{rook_start}2", f"{rook_start}1", "CLOSE_GRIPPER", f"{rook_start}2", f"{rook_end}2", f"{rook_end}1", "OPEN_GRIPPER", f"{rook_end}2",
            "HOME"
        ]

    start = input_move[:2]
    end = input_move[2:]
    if capture:
        return [
            f"{end}2", f"{end}1", "CLOSE_GRIPPER", f"{end}2", "OUT", "OPEN_GRIPPER",
            f"{start}2", f"{start}1", "CLOSE_GRIPPER", f"{start}2",
            f"{end}2", f"{end}1", "OPEN_GRIPPER", f"{end}2", "HOME"
        ]
    else:
        return [
            f"{start}2", f"{start}1", "CLOSE_GRIPPER", f"{start}2",
            f"{end}2", f"{end}1", "OPEN_GRIPPER", f"{end}2", "HOME"
        ]

def initialize_stockfish_board():
    """
    Initialize the Stockfish board and return the current board state.
    """
    stockfish.set_position([])  # Reset to the initial chessboard state
    return stockfish.get_board_visual()

def display_stockfish_board(is_white_turn):
    """
    Display the current board state using Stockfish's visual representation and
    print whose turn it is.
    """
    print("\nCurrent Board:")
    print(stockfish.get_board_visual())
    if is_white_turn:
        print("It's Black's turn.")
    else:
        print("It's White's turn.")

def detect_changes(previous_squares, current_squares, threshold=4):
    changes = []
    for square_name, current_square in current_squares.items():
        previous_square = previous_squares[square_name]
        current_gray = cv2.cvtColor(current_square, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous_square, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(current_gray, previous_gray)
        mean_diff = cv2.mean(diff)[0]
        if mean_diff > threshold:
            changes.append((square_name, mean_diff))
    changes.sort(key=lambda x: x[1], reverse=True)
    return changes

def enhance_br_contrast(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = cv2.add(h, 50)  # Increase saturation
    v = cv2.add(v, 60)  # Increase saturation
    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    return enhanced_image

def fen_to_board(fen):
    """
    Convert a FEN string to a board representation as a dictionary, including empty squares.

    Args:
        fen (str): The FEN string of the board.

    Returns:
        dict: A dictionary mapping square names (e.g., 'e4') to piece types (e.g., 'P', 'r') or None for empty squares.
    """
    board = {}
    rows = fen.split(' ')[0].split('/')
    for rank_index, row in enumerate(rows):
        file_index = 0
        for char in row:
            if char.isdigit():
                # Fill empty squares with None
                for _ in range(int(char)):
                    square = f"{chr(ord('a') + file_index)}{8 - rank_index}"
                    board[square] = None
                    file_index += 1
            else:
                # Add the piece to the square
                square = f"{chr(ord('a') + file_index)}{8 - rank_index}"
                board[square] = char
                file_index += 1
    return board

def classify_stockfish_move(move):
    """
    Classify a Stockfish move as normal, capture, castling, promotion, or en passant.

    Args:
        move (str): Stockfish move in standard algebraic notation (e.g., 'e2e4', 'e1g1', 'a7a8q').

    Returns:
        str: The type of the move ('normal', 'capture', 'castling', 'promotion', 'en passant').
    """
    print(f"\nClassifying move: {move}")

    # Castling moves
    if move in ["e1g1", "e1c1", "e8g8", "e8c8"]:
        print("Detected castling move.")
        return "castling"

    # Detect promotion
    if len(move) == 5 and move[4] in "qrbn":  # e.g., a7a8q
        print("Detected promotion move.")
        return "promotion"

    # Get source and destination squares
    source_square = move[:2]
    destination_square = move[2:4]
    print(f"Source square: {source_square}, Destination square: {destination_square}")

    # Get initial and updated boards
    initial_fen = stockfish.get_fen_position()
    initial_board = fen_to_board(initial_fen)
    print(f"Initial board: {initial_board}")

    stockfish.make_moves_from_current_position([move])
    updated_fen = stockfish.get_fen_position()
    updated_board = fen_to_board(updated_fen)
    print(f"Updated board: {updated_board}")

    # Restore the initial position to avoid disrupting the game state
    stockfish.set_fen_position(initial_fen)

    # Detect if a piece was captured
    captured_pieces = {
        square: piece
        for square, piece in initial_board.items()
        if square != source_square and  # Exclude source square
           (square not in updated_board or updated_board.get(square) != piece)
    }
    print(f"Captured pieces (before filtering): {captured_pieces}")

    # Filter out cases where the destination square was empty before the move
    if destination_square in captured_pieces:
        if initial_board[destination_square] is None:
            print(f"Removing {destination_square} from captured pieces because it was empty.")
            captured_pieces.pop(destination_square)

    print(f"Captured pieces (after filtering): {captured_pieces}")

    if captured_pieces:
        # Detect en passant
        initial_ep_square = initial_fen.split(' ')[3]  # En passant square in FEN
        print(f"En passant square: {initial_ep_square}")
        if initial_ep_square != '-' and destination_square == initial_ep_square:
            print("Detected en passant capture.")
            return "en passant"
        print("Detected capture.")
        return "capture"

    # If no capture detected, it's a normal move
    print("Detected normal move.")
    return "normal"

def undo_last_move():
    """Undo the last move by resetting Stockfish to the previous state."""
    if move_history:
        last_move = move_history.pop()  # Remove last move
        stockfish.set_position(move_history)  # Reset Stockfish position
        is_white_turn = True
        display_stockfish_board(is_white_turn)
        print(f"\nUndoing move: {last_move}")
    else:
        print("\nNo moves to undo.")

# Function to draw buttons inside the sidebar
def draw_buttons(frame, sidebar):
    for button in buttons:
        x1, y1, x2, y2 = button["pos"]
        color = button["color"]
        cv2.rectangle(sidebar, (x1, y1), (x2, y2), color, -1)  # Filled button
        cv2.putText(sidebar, button["label"], (x1 + 10, y1 + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)

# Mouse click event function
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > frame_width:  # Sidebar starts after the image width
            for button in buttons:
                x1, y1, x2, y2 = button["pos"]
                if x1 < (x - frame_width) < x2 and y1 < y < y2:  # Adjust for sidebar position
                    print(f"Clicked {button['label']} (simulating key '{button['key']}')")
                    pyautogui.press(button["key"])  # Simulate key press


def process_chessboard(camera_index=4, projection_output_dir="projected_images/", chess_output_dir="chess_squares/", stockfish_move=None, move_type=None, base_client=None):
    """
    Process the physical chessboard to detect changes, capture user's move with 'c',
    trigger Stockfish's move with 's', and execute Stockfish's move with the robotic arm using 'm'.
    """
    os.makedirs(projection_output_dir, exist_ok=True)
    os.makedirs(chess_output_dir, exist_ok=True)

    global frame_width  # Store width for sidebar placement
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Camera {camera_index} could not be accessed.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        return

    frame_height, frame_width = frame.shape[:2]  # Get frame size
    sidebar_width = 330  # Sidebar width for buttons
    window_width = frame_width + sidebar_width  # Total window width

    cv2.namedWindow("Chess Game")
    cv2.setMouseCallback("Chess Game", on_mouse)  # Enable mouse clicks

    # Ask for the player's color
    player_color = input("Do you want to play as White (w) or Black (b)? ").strip().lower()
    if player_color == 'w':
        print("You are playing as White.")
        user_is_white = True
    else:
        print("You are playing as Black.")
        user_is_white = False

    # Original logic for the board processing
    src_points = np.array([[82, 13], [55, 452], [520, 455], [503, 13]], dtype="float32")
    width, height = 500, 450
    dst_points = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    previous_squares = None
    image_counter = 1
    is_white_turn = user_is_white

    print("Press 'c' to capture your move, 's' for Stockfish's move, 'm' to execute Stockfish's move with the arm, or 'q' to quit.")

    if not user_is_white:
        stockfish_move = stockfish.get_best_move()
        stockfish.make_moves_from_current_position([stockfish_move])
        move_history.append(stockfish_move)  # Store move in history
        print(f"move history: {move_history}")
        is_white_turn = True
        display_stockfish_board(is_white_turn)
        print(f"Executing Stockfish's move: {stockfish_move}")

        move_sequence = get_movement_sequence(stockfish_move, 
                                                capture=(move_type == "capture"),
                                                castling=(move_type == "castling"),
                                                promotion=(move_type == "promotion"))
        for step in move_sequence:
            if step == "CLOSE_GRIPPER":
                set_gripper_position(base_client, 0.99)  # Close gripper
            elif step == "OPEN_GRIPPER":
                set_gripper_position(base_client, 0.6)  # Open gripper
            else:
                position = POSITIONS_MAP.get(step)
                if position:
                    success = set_actuator_positions_directly(base_client, position)
                    if not success:
                        print(f"Failed to move to position: {step}")
                        break
                else:
                    print(f"Position {step} not found in CSV file.")
        print("Move execution completed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Create a sidebar to hold buttons
        sidebar = np.full((frame_height, sidebar_width, 3), (80, 80, 20), dtype=np.uint8)

        # Draw buttons in the sidebar
        draw_buttons(frame, sidebar)

        # Combine the image and sidebar
        combined_frame = np.hstack((frame, sidebar))

        # Show the combined output
        cv2.imshow("Chess Game", combined_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Capture and process the board
            projected = cv2.warpPerspective(frame, matrix, (width, height))
            enhanced_projected = enhance_br_contrast(projected)

            cropped = enhanced_projected[19:433, 21:481]
            square_height, square_width = cropped.shape[0] // 8, cropped.shape[1] // 8

            current_squares = {}
            for row in range(8):
                for col in range(8):
                    x_start, y_start = col * square_width, row * square_height
                    x_end, y_end = x_start + square_width, y_start + square_height
                    square = cropped[y_start:y_end, x_start:x_end]
                    square_name = f"{chr(ord('a') + col)}{8 - row}"
                    square_path = os.path.join(chess_output_dir, f"{square_name}_img_{image_counter:03d}.jpg")
                    cv2.imwrite(square_path, square)
                    current_squares[square_name] = square

            print(f"Image {image_counter} captured and processed.")

            if previous_squares:
                changes = detect_changes(previous_squares, current_squares)
                if len(changes) >= 2:
                    from_square, to_square = changes[0][0], changes[1][0]
                    detected_move = f"{from_square}{to_square}"

                    if stockfish.is_move_correct(detected_move):
                        stockfish.make_moves_from_current_position([detected_move])
                        move_history.append(detected_move)  # Store move in history
                        print(f"move history: {move_history}")
                        display_stockfish_board(is_white_turn)
                        print(f"Your move detected and validated: {detected_move}")
                        is_white_turn = not is_white_turn  # Change turn
                    else:
                        reversed_move = f"{to_square}{from_square}"
                        if stockfish.is_move_correct(reversed_move):
                            stockfish.make_moves_from_current_position([reversed_move])
                            move_history.append(reversed_move)  # Store move in history
                            print(f"move history: {move_history}")
                            display_stockfish_board(is_white_turn)
                            print(f"Your move detected and validated (reversed): {reversed_move}")
                            is_white_turn = not is_white_turn  # Change turn
                        else:
                            print(f"Invalid move detected: {detected_move} or {reversed_move}")

                else:
                    print("No significant changes detected. Turn does not change.")
            
            previous_squares = current_squares
            image_counter += 1

        elif key == ord('s'):
            # Stockfish's turn
            stockfish_move = stockfish.get_best_move()
            if stockfish_move:
                move_type = classify_stockfish_move(stockfish_move)
                stockfish.make_moves_from_current_position([stockfish_move])
                move_history.append(stockfish_move)  # Store move in history
                print(f"move history: {move_history}")
                is_white_turn = True
                display_stockfish_board(is_white_turn)
                print(f"Stockfish's move: {stockfish_move} ({move_type})")
            else:
                print("Game over!")
                break

        elif key == ord('m') and stockfish_move:
            # Move Stockfish's piece using the robotic arm
            print(f"Executing Stockfish's move: {stockfish_move}")
            move_sequence = get_movement_sequence(stockfish_move, 
                                                   capture=(move_type == "capture"),
                                                   castling=(move_type == "castling"),
                                                   promotion=(move_type == "promotion"))
            for step in move_sequence:
                if step == "CLOSE_GRIPPER":
                    set_gripper_position(base_client, 0.99)  # Close gripper
                elif step == "OPEN_GRIPPER":
                    set_gripper_position(base_client, 0.6)  # Open gripper
                else:
                    position = POSITIONS_MAP.get(step)
                    if position:
                        success = set_actuator_positions_directly(base_client, position)
                        if not success:
                            print(f"Failed to move to position: {step}")
                            break
                    else:
                        print(f"Position {step} not found in CSV file.")
            print("Move execution completed.")

        elif key == ord('p'):
            # Stockfish's turn
            stockfish_move = stockfish.get_best_move()
            if stockfish_move:
                move_type = classify_stockfish_move(stockfish_move)
                stockfish.make_moves_from_current_position([stockfish_move])
                move_history.append(stockfish_move)  # Store move in history
                print(f"move history: {move_history}")
                is_white_turn = True
                display_stockfish_board(is_white_turn)
                print(f"Stockfish's move: {stockfish_move} ({move_type})")
            else:
                print("Game over!")
                break
            # Move Stockfish's piece using the robotic arm
            print(f"Executing Stockfish's move: {stockfish_move}")
            move_sequence = get_movement_sequence(stockfish_move, 
                                                   capture=(move_type == "capture"),
                                                   castling=(move_type == "castling"),
                                                   promotion=(move_type == "promotion"))
            for step in move_sequence:
                if step == "CLOSE_GRIPPER":
                    set_gripper_position(base_client, 0.99)  # Close gripper
                elif step == "OPEN_GRIPPER":
                    set_gripper_position(base_client, 0.6)  # Open gripper
                else:
                    position = POSITIONS_MAP.get(step)
                    if position:
                        success = set_actuator_positions_directly(base_client, position)
                        if not success:
                            print(f"Failed to move to position: {step}")
                            break
                    else:
                        print(f"Position {step} not found in CSV file.")
            print("Move execution completed.")

        elif key == ord('u'):
            undo_last_move()

        elif key == ord('t'):
            # Manually input a move through the terminal
            manual_move = input("Enter your move (e.g., e2e4): ").strip().lower()

            # Validate move using Stockfish
            if stockfish.is_move_correct(manual_move):
                stockfish.make_moves_from_current_position([manual_move])
                move_history.append(manual_move)  # Store move in history
                display_stockfish_board(is_white_turn)
                print(f"Your move has been manually added: {manual_move}")
                is_white_turn = not is_white_turn  # Change turn
            else:
                print(f"Invalid move: {manual_move}. Please try again.")

        elif key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    args = type('Args', (object,), {"ip": ROBOT_IP, "username": USERNAME, "password": PASSWORD})()

    # Initialize the board with Stockfish and display it
    initialize_stockfish_board()
    display_stockfish_board(is_white_turn=True)  # Assuming it's White's turn initially

    # Start image processing for the physical board
    print("Press 'c' to capture and compare your move, 's' for Stockfish's move, 'm' for the robotic arm to execute Stockfish's move, or 'q' to quit.")

    stockfish_move = None
    move_type = None  # To store the type of Stockfish move

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        try:
            base_client = BaseClient(router)

            # Use the original `process_chessboard` function's camera logic
            process_chessboard(
                stockfish_move=stockfish_move,
                move_type=move_type,
                base_client=base_client
            )

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
