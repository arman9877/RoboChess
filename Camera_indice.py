import cv2

def find_camera_index(max_index=10):
    """
    Find available camera indices by testing up to a specified maximum index.

    Args:
        max_index (int): The maximum camera index to check.

    Returns:
        list: A list of available camera indices.
    """
    available_cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

# Check camera indices
max_index_to_check = 10  # You can increase this if needed
available_indices = find_camera_index(max_index_to_check)

if available_indices:
    print(f"Available camera indices: {available_indices}")
else:
    print("No cameras found.")
