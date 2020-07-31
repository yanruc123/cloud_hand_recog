from keras.models import model_from_json
import numpy as np
from skimage import io
import cv2
import random

def judge(u, c):
    """
    0:user win
    1:computer win
    2:draw
    """
    if u == "Rock":
        if c == "Scissor":
            return "computer win"
        if c == "Rock":
            return "draw"
        elif c == "Paper":
            return "user win"
    elif u == "Scissor":
        if c == "Scissor":
            return "draw"
        if c == "Rock":
            return "computer win"
        elif c == "Paper":
            return "user win"
    else:  # u == "Paper"
        if c == "Scissor":
            return "computer win"
        if c == "Rock":
            return "user win"
        elif c == "Paper":
            return "draw"

def prepImg(pth):
    return cv2.resize(pth,(150,150)).reshape(1,150,150,3)


shape_to_label = {'Rock':np.array([1.,0.,0.]),'Paper':np.array([0.,1.,0.]),'Scissor':np.array([0.,0.,1.])}
arr_to_shape = {np.argmax(shape_to_label[x]):x for x in shape_to_label.keys()}

from keras.models import load_model
loaded_model = load_model('rps.h5')

shape_to_label = {'Rock': np.array([1., 0., 0.]), 'Paper': np.array([0., 1., 0.]), 'Scissor': np.array([0., 0., 1.])}
arr_to_shape = {np.argmax(shape_to_label[x]): x for x in shape_to_label.keys()}

#options = ['Rock', 'Paper', 'Scissor']
#winRule = {'Rock': 'Scissor', 'Scissor': 'Paper', 'Paper': 'Rock'}
rounds = 0
compScore = 0
playerScore = 0
computer_give = ""

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
loaded_model.predict(prepImg(frame[50:350, 100:400]))

bplay = ""

#一次过后所有数值清零
compScore = 0
playerScore = 0

def winner(c,p):
    if (c > p):
        win = "Computer"
    elif (c < p):
        win = "Player"
    else:
        win = "Draw"

while True:
    ret, frame = cap.read()
    frame = frame = cv2.putText(frame, "Press Space to start", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (126, 255, 255),
                                2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor', frame)
    if cv2.waitKey(1) & 0xff == ord(' '):
        break

for rounds in range(5):
    pred = ""
    for i in range(90):
        ret, frame = cap.read()

        # Countdown
        if i // 20 < 3:
            frame = cv2.putText(frame, str(i // 20 + 1), (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (126, 255, 255), 2,
                                cv2.LINE_AA)

        # Prediction
        elif i / 20 < 3.5:
            pred = arr_to_shape[np.argmax(loaded_model.predict(prepImg(frame[50:350, 100:400])))]
            print("pred = ", pred)

        # Get Bots Move
        elif i / 20 == 3.5:
            computer_choice = ["Rock", "Paper", "Scissor"]
            computer_give = random.choice(computer_choice)
            print("computer_give =", computer_give)

            #then judge
            result = judge(pred,computer_give)
            print("result = ", result)
            if (result == "computer win"):
                compScore += 1
            elif (result == "user win"):
                playerScore += 1

        # Update Score
        elif i // 20 == 4:
            #playerScore, compScore = updateScore(pred, bplay, playerScore, compScore)
            break

        #print(playerScore,compScore)

        cv2.rectangle(frame, (100, 150), (300, 350), (255, 255, 255), 2)
        frame = cv2.putText(frame, "Player : {}      Computer : {}".format(playerScore, compScore), (120, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (126, 255, 255), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, pred, (150, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 250, 0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Computer Played : {}".format(computer_give), (300, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (126, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissor', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break


while True:
    ret, frame = cap.read()
    frame = cv2.putText(frame, winner(compScore, playerScore), (230, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (126, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "Press q to quit", (190, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (126, 255, 255), 2,
                        cv2.LINE_AA)
    frame = cv2.putText(frame, "Player : {}      Bot : {}".format(playerScore, compScore), (120, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (126, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissor', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
