import sys
import os
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
    "Skill Level": 15,
})

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
    time.sleep(0.7)  # Allow gripper to complete the movement

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
            f"{end}2", f"{end}1", "OPEN_GRIPPER", f"{end}2"
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
            f"{rook_start}2", f"{rook_start}1", "CLOSE_GRIPPER", f"{rook_start}2", f"{rook_end}2", f"{rook_end}1", "OPEN_GRIPPER", f"{rook_end}2"
        ]

    start = input_move[:2]
    end = input_move[2:]
    if capture:
        return [
            f"{end}2", f"{end}1", "CLOSE_GRIPPER", f"{end}2", "OUT", "OPEN_GRIPPER",
            f"{start}2", f"{start}1", "CLOSE_GRIPPER", f"{start}2",
            f"{end}2", f"{end}1", "OPEN_GRIPPER", f"{end}2"
        ]
    else:
        return [
            f"{start}2", f"{start}1", "CLOSE_GRIPPER", f"{start}2",
            f"{end}2", f"{end}1", "OPEN_GRIPPER", f"{end}2"
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
        print("It's White's turn.")
    else:
        print("It's Black's turn.")

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

def enhance_green_contrast(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 50)  # Increase saturation
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

def process_chessboard(base_client):
    """
    Automatically play chess using Stockfish and control the robotic arm.
    """
    initialize_stockfish_board()  # Set up Stockfish
    is_white_turn = True  # Start with White's turn

    while True:
        # Get Stockfish's move
        stockfish_move = stockfish.get_best_move()
        if not stockfish_move:
            print("Game over!")
            break

        move_type = classify_stockfish_move(stockfish_move)
        print(f"Stockfish's move: {stockfish_move} ({move_type})")

        stockfish.make_moves_from_current_position([stockfish_move])
        display_stockfish_board(is_white_turn)

        # Execute the move with the robot arm
        print(f"Executing Stockfish's move: {stockfish_move}")
        move_sequence = get_movement_sequence(
            stockfish_move,
            capture=(move_type == "capture"),
            castling=(move_type == "castling"),
            promotion=(move_type == "promotion")
        )

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

        # Wait for 1 second before the next turn
        time.sleep(0.7)
        is_white_turn = not is_white_turn

def main():
    args = type('Args', (object,), {"ip": ROBOT_IP, "username": USERNAME, "password": PASSWORD})()

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        try:
            base_client = BaseClient(router)
            process_chessboard(base_client)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
