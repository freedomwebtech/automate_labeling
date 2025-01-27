
import cv2    
import time
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the images folder path (relative to the script's location)
folder_path = os.path.join(script_dir, "images")

# Create the images folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

cpt = 0
maxFrames = 50  # You want a maximum of 50 frames

count = 0
cap = cv2.VideoCapture('helmet.avi')  # Open the webcam

while cpt < maxFrames:
    ret, frame = cap.read()  # Capture a frame
    if not ret:
        break  # If frame capture fails, break the loop
    count += 1
    if count % 3 != 0:  # Only capture every third frame
        continue
    frame = cv2.resize(frame, (1080, 500))  # Resize the frame

    # Show the image in a window
    cv2.imshow("test window", frame)  

    # Save the image to the images folder (relative to the script's directory)
    image_path = os.path.join(folder_path, f"img_{cpt}.jpg")
    cv2.imwrite(image_path, frame)
    
    # Sleep for 0.01 seconds between frames to avoid overloading
    time.sleep(0.01)
    
    # Increment the frame counter
    cpt += 1
    
    # Break the loop if the 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the capture and close windows
cap.release()   
cv2.destroyAllWindows()
