import cv2
import numpy as np

# Define the first quadrilateral (source points) from the live camera frame
src_points = np.array([[82, 13], [55, 452], [520, 455], [503, 13]], dtype="float32")

# Define destination points for warping (rectangular perspective)
width, height = 500, 450
dst_points = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype="float32")

# Define the new smaller square inside the quadrilateral
inner_square = np.array([[22, 19], [20, 433], [482, 432], [481, 18]], dtype="float32")

# Open camera
cap = cv2.VideoCapture(4)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Compute the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective warp to get a top-down view
    warped = cv2.warpPerspective(frame, matrix, (width, height))

    # Create a blank white frame to display the transformed quadrilateral
    new_frame = np.ones((height, width, 3), dtype=np.uint8) * 255  

    # Place the warped region inside the new frame
    new_frame[0:height, 0:width] = warped

    # Draw the inner square inside the new frame (green lines, cyan dots)
    for i in range(len(inner_square)):
        cv2.line(new_frame, tuple(inner_square[i].astype(int)), 
                 tuple(inner_square[(i + 1) % len(inner_square)].astype(int)), (0, 255, 0), 1)
        cv2.circle(new_frame, tuple(inner_square[i].astype(int)), 3, (255, 255, 0), -1)

    # Draw the original quadrilateral on the live frame (green lines, red dots)
    for i in range(len(src_points)):
        cv2.line(frame, tuple(src_points[i].astype(int)), 
                 tuple(src_points[(i + 1) % len(src_points)].astype(int)), (0, 255, 0), 1)
        cv2.circle(frame, tuple(src_points[i].astype(int)), 3, (0, 0, 255), -1)

    # Display both frames
    cv2.imshow('Original Camera Frame', frame)
    cv2.imshow('Warped Quadrilateral Inside New Frame', new_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
