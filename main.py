import mediapipe as mp
import cv2
import numpy as np
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# function to calculate angle of 2 vectors
def angle_of_vec(v1, v2):
    try:
        angle= math.degrees(math.acos((v1[0]*v2[0] + v1[1]*v2[1]) / (( (v1[0]**2 + v1[1]**2)**0.5) * ( (v2[0]**2 + v2[1]**2)**0.5) )))
    except:
        angle = 180
    return angle

def closed_fingers(point):
    angles = []
    for i in range(1, 18, 4):
        v1 = [point[0].x - point[i+1].x, point[0].y - point[i+1].y]
        v2 = [point[i+3].x - point[i+2].x, point[i+3].y - point[i+2].y]
        angles.append(angle_of_vec(v1, v2))
    # index finger is closed
    closed = []
    for i in range(5):
        print(i, angles[i])
        if angles[i] < 135:
            closed.append(i)
    
    return closed
        
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    # miror the frame and convert bgr to rgb
    # frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    results = hands.process(frame)
    
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks( frame, landmarks, mp_hands.HAND_CONNECTIONS)
        
            closed = closed_fingers(landmarks.landmark)
            
            if closed == [0, 2, 3, 4]:
                cv2.putText(frame, text='1', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            elif closed == [0, 3, 4]:
                cv2.putText(frame, text='2', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            elif closed == [0, 4]:
                cv2.putText(frame, text='3', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            elif closed == [0]:
                cv2.putText(frame, text='4', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            elif closed == []:
                cv2.putText(frame, text='5', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            elif closed == [1, 2, 3]:
                cv2.putText(frame, text='6', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            elif closed == [2, 3, 4]:
                cv2.putText(frame, text='7', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            elif closed == [3, 4]:
                cv2.putText(frame, text='8', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            elif closed == [4]:
                cv2.putText(frame, text='9', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            elif closed == [0, 1, 2, 3, 4]:
                cv2.putText(frame, text='0', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            # elif closed == [0, 1, 3, 4]:
            #     cv2.putText(frame, text='Fuck', org=(200, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale=3, lineType=cv2.LINE_AA)
            elif closed == [1]:
                cv2.putText(frame, text='Okay', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
            elif closed == [2,3]:
                cv2.putText(frame, text='Rock', org=(300, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=2, lineType=cv2.LINE_AA)
    
    cv2.imshow('Gesture Recognition', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

