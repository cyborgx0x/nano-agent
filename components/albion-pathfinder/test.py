import numpy as np
import PIL.ImageGrab
import cv2
import time

# Create the window for displaying the screen stream
cv2.namedWindow("Screen Stream")

# Initialize the timer
start_time = time.time()
frame_count = 0

while True:
    # Capture the screen as a PIL image
    pil_image = PIL.ImageGrab.grab()

    # Convert the PIL image to a numpy array
    np_array = np.array(pil_image)

    # Convert the color format from RGB to BGR
    bgr_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)

    # Display the captured image in the window with FPS in title
    cv2.imshow("Screen Stream", bgr_array)
    cv2.setWindowTitle("Screen Stream", f"Screen Stream ({int(frame_count/(time.time()-start_time))} FPS)")

    # Increment the frame count and update the timer
    frame_count += 1

    # Wait for a key press and check if it is the 'q' key
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()