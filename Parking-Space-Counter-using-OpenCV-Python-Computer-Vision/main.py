import cv2
import pickle
import cvzone
import numpy as np

# Load video
cap = cv2.VideoCapture('carPark.mp4')

# Load saved parking positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48

def checkParkingSpaces(imgPro):
    spaceCounter = 0
    
    for pos in posList:
        x, y = pos
        
        # Crop the image for each parking space
        imgCrop = imgPro[y:y+height, x:x+width]
        
        # Count white pixels in the cropped image
        count = cv2.countNonZero(imgCrop)
        
        # Display the count on the image
        cvzone.putTextRect(img, str(count), (x, y+height-3), scale=1.5, thickness=2, offset=0)
        
        # Define threshold for empty space
        threshold = 800
        
        # Color logic: Green for empty, Red for occupied
        if count < threshold:
            color = (0, 255, 0)  # Green for empty spaces
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)  # Red for occupied spaces
            thickness = 2
            
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), color, thickness)
    
    # Display total available spaces
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0,200,0))

# Main loop
while True:
    # Reset video when it ends
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    success, img = cap.read()
    if not success:
        break
        
    # Image processing pipeline
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    
    # Check parking spaces
    checkParkingSpaces(imgDilate)
    
    # Display the processed image
    cv2.imshow("Image", img)
    

    cv2.waitKey(10) 
