import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
import os

# Set console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Static window size variables
winSizeX = 640
winSizeY = 480

# Initialize board color
BoundryInc = 5
White = (255, 255, 255)
Black = (0, 0, 0)
Red = (255, 0, 0)

imageSave = False

# Load the trained model
MODEL = load_model("trainModel.keras")

# Dictionary for labels
LABELS = {0: "Zero", 1: "One", 2: "Two",
          3: "Three", 4: "Four", 5: "Five",
          6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Initialize Pygame
pygame.init()

# Load font
FONT = pygame.font.Font("freesansbold.ttf", 18)

# Set up display surface
displaySurf = pygame.display.set_mode((winSizeX, winSizeY))
pygame.display.set_caption("Digit Recognition")

# Event loop
iswriting = False
number_Xcord = []
number_Ycord = []

imageCount = 1
Predict = True

# Keep track of bounding boxes and labels
digit_rects = []
digit_labels = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and iswriting:
            xCord, yCord = event.pos
            pygame.draw.circle(displaySurf, White, (xCord, yCord), 4, 0)

            number_Xcord.append(xCord)
            number_Ycord.append(yCord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if number_Xcord and number_Ycord:  # Check if the lists are not empty
                number_Xcord = sorted(number_Xcord)
                number_Ycord = sorted(number_Ycord)

                rectMinX, rectMaxX = max(number_Xcord[0] - BoundryInc, 0), min(winSizeX, number_Xcord[-1] + BoundryInc)
                rectMinY, rectMaxY = max(number_Ycord[0] - BoundryInc, 0), min(winSizeY, number_Ycord[-1] + BoundryInc)

                img_Arr = np.array(pygame.PixelArray(displaySurf))[rectMinX:rectMaxX, rectMinY:rectMaxY].T.astype(np.float32)

                if imageSave:
                    cv2.imwrite(f"img_{imageCount}.png", img_Arr)
                    imageCount += 1
                
                if Predict:
                    img = cv2.resize(img_Arr, (28, 28))
                    img = np.pad(img, (10, 10), 'constant', constant_values=0)
                    img = cv2.resize(img, (28, 28)) / 255.0

                    label = str(LABELS[np.argmax(MODEL.predict(img.reshape(1, 28, 28, 1)))])

                    # Store the bounding box and the label
                    digit_rects.append((rectMinX, rectMaxY))
                    digit_labels.append(label)
            
            # Reset coordinates for the next digit
            number_Xcord = []
            number_Ycord = []
        
        if event.type == KEYDOWN:
            if event.unicode == "n":
                displaySurf.fill(Black)
                digit_rects = []  # Clear stored bounding boxes
                digit_labels = []  # Clear stored labels
    
    # Display all predicted labels
    for (x, y), label in zip(digit_rects, digit_labels):
        textSurf = FONT.render(label, True, Red, White)
        textRect = textSurf.get_rect()
        textRect.left, textRect.bottom = x, y

        displaySurf.blit(textSurf, textRect)
    
    pygame.display.update()
