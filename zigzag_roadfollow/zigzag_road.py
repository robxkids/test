import cv2
import numpy as np

video = cv2.VideoCapture('outpy.avi')

while(video.isOpened()):
    ret, frame = video.read()
    if ret:
        #желтая линия разметки: синий(0 - 130), зеленый (130 - 255), красный (140 - 255)
        #белая линия разметки: синий(140 - 255), зеленый (120 - 255), красный (120 - 255)
        binary_yellow = cv2.inRange(frame, (0, 130, 140), (130, 255, 255))
        binary_white = cv2.inRange(frame, (140, 120, 120), (255, 255, 255))

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        src = np.int32([[0, frame_height - 50], [50, frame_height/3], [frame_width - 50, frame_height/3], [frame_width, frame_height - 50]])
        src_float = np.array(src, dtype=np.float32)
        cv2.polylines(frame, [src], True, (255, 0, 0), 5)

        dst_height = int((frame_height - 50 - frame_height/3)/2)
        dst_width = int(frame_width/2)
        dst = np.float32([[0, dst_height], [0, 0], [dst_width, 0], [dst_width, dst_height]])
        M = cv2.getPerspectiveTransform(src_float, dst)

        warped_yellow = cv2.warpPerspective(binary_yellow, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR)
        warped_white = cv2.warpPerspective(binary_white, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR)

        out_yellow = np.dstack((warped_yellow, warped_yellow, warped_yellow))
        out_white = np.dstack((warped_white, warped_white, warped_white))
        out_img = out_white + out_yellow

        histogram_yellow = np.sum(warped_yellow[:,:], axis=0) #histogram_yellow = [1520, 0, 255 ...  512] len(histogram_yellow) = 400
        midpoint = histogram_yellow.shape[0] // 2 # 200
        indWhiteColumnL = np.argmax(histogram_yellow[:midpoint]) #индекс элемента с наибольшим значением
        warped_yellow_visual = warped_yellow.copy()
        cv2.line(out_img, (indWhiteColumnL, 0), (indWhiteColumnL, histogram_yellow.shape[0]), (0, 255, 255), 1)

        histogram_white = np.sum(warped_white[:,:], axis=0)
        midpoint2 = histogram_white.shape[0] // 2
        indWhiteColumnR = np.argmax(histogram_white[midpoint2:]) + midpoint2
        warped_white_visual = warped_white.copy()
        cv2.line(out_img, (indWhiteColumnR, 0), (indWhiteColumnR, histogram_white.shape[0]), (255, 255, 255), 1)

        nwindows = 9
        window_height = out_img.shape[0] // 9
        window_half_width = 20
        XCenterLeftWindow = indWhiteColumnL
        XCenterRightWindow = indWhiteColumnR

        for window in range(nwindows):
            win_y1 = out_img.shape[0] - (window + 1) * window_height
            win_y2 = out_img.shape[0] - (window) * window_height
            left_win_x1 = XCenterLeftWindow - window_half_width
            left_win_x2 = XCenterLeftWindow + window_half_width
            right_win_x1 = XCenterRightWindow - window_half_width
            right_win_x2 = XCenterRightWindow + window_half_width

            cv2.rectangle(out_img, (left_win_x1, win_y1), (left_win_x2, win_y2), (0, 255, 255), 1)


        cv2.imshow("frame", frame)
        cv2.imshow("warped_yellow", warped_yellow_visual)
        cv2.imshow("warped_white", warped_white_visual)
        cv2.imshow("out", out_img)

    else:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('screen.png', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()