import cv2
import os
import numpy as np

# Connect to the zero-latency Pi stream
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|analyzeduration;0|probesize;32"
cap = cv2.VideoCapture("rtsp://100.91.4.6:8554/cam", cv2.CAP_FFMPEG)

print("Connecting to camera...")
ret, frame = cap.read()
if not ret:
    print("Failed to grab a frame. Make sure the Pi stream is running.")
    exit()

points = []

def map_polygon(event, x, y, flags, params):
    # On left mouse click, record the coordinates and draw a marker
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        # Draw a line connecting to the previous point
        if len(points) >= 2:
            cv2.line(frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 
0), 2)
            
        cv2.imshow("Zone Mapper - Click to Draw, Press 'q' to Finish", 
frame)

cv2.imshow("Zone Mapper - Click to Draw, Press 'q' to Finish", frame)
cv2.setMouseCallback("Zone Mapper - Click to Draw, Press 'q' to Finish", 
map_polygon)

cv2.waitKey(0)

# Once 'q' is pressed, print the exact Python code needed for the main 
print("\n--- COPY AND PASTE THIS LINE INTO pose_tracker.py ---")
print(f"restricted_zone = np.array({points}, np.int32)")
print("-----------------------------------------------------\n")

cv2.destroyAllWindows()
cap.release()
