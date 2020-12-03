import cv2
import numpy

def nothing1(arg):
    pass

video = cv2.VideoCapture('outpy.avi')
cv2.namedWindow('setup') #создаем именованное окно setup
cv2.createTrackbar('b1', 'setup', 0, 255, nothing1) #создаем трекбар в окне от 0 до 255
cv2.createTrackbar('g1', 'setup', 0, 255, nothing1)
cv2.createTrackbar('r1', 'setup', 0, 255, nothing1)
cv2.createTrackbar('b2', 'setup', 255, 255, nothing1)
cv2.createTrackbar('g2', 'setup', 255, 255, nothing1)
cv2.createTrackbar('r2', 'setup', 255, 255, nothing1)

while(video.isOpened()):
    ret, frame = video.read()
    if ret:

        b1 = cv2.getTrackbarPos('b1', 'setup')
        g1 = cv2.getTrackbarPos('g1', 'setup')
        r1 = cv2.getTrackbarPos('r1', 'setup')
        b2 = cv2.getTrackbarPos('b2', 'setup')
        g2 = cv2.getTrackbarPos('g2', 'setup')
        r2 = cv2.getTrackbarPos('r2', 'setup')
        # print(b1)

        inRange = cv2.inRange(frame, (b1, g1, r1), (b2, g2, r2))

        cv2.imshow('setup', inRange)
        cv2.imshow('color', frame)


    else:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('screen.png', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()