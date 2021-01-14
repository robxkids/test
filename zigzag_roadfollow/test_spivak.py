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
        src = np.int32([[0, frame_height - 200], [100, frame_height/3], [frame_width - 100, frame_height/3], [frame_width, frame_height - 200]])
        src_float = np.array(src, dtype=np.float32)
        cv2.polylines(frame, [src], True, (255, 0, 0), 5)

        dst_height = int((frame_height - 50 - frame_height/3)/2)
        dst_width = int(frame_width/2)
        dst = np.float32([[0, dst_height], [0, 0], [dst_width, 0], [dst_width, dst_height]])
        M = cv2.getPerspectiveTransform(src_float, dst)
        warped_yellow = cv2.warpPerspective(binary_yellow, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR)
        warped_white = cv2.warpPerspective(binary_white, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR)

        #строим гистограмму, максимальная сумма всех пикселей по вертикальной оси, в левой части изображения,
        #считаем только с середины картинки по высоте
        histogram_yellow = np.sum(warped_yellow[: warped_yellow.shape[0], :], axis=0)
        midpoint = histogram_yellow.shape[0] // 2
        indWhiteColumnL = np.argmax(histogram_yellow[:])
        warped_yellow_visual = warped_yellow.copy()
        cv2.line(warped_yellow_visual, (indWhiteColumnL, 0), (indWhiteColumnL, histogram_yellow.shape[0]), 255, 5)

        histogram_white = np.sum(warped_white[: warped_white.shape[0], :], axis=0)
        midpoint = histogram_white.shape[0] // 2
        indWhiteColumnR = np.argmax(histogram_white[midpoint:]) + midpoint
        warped_white_visual = warped_white.copy()
        cv2.line(warped_white_visual, (indWhiteColumnR, 0), (indWhiteColumnR, histogram_white.shape[0]), 255, 5)

        nwindows = 9
        window_height = np.int(dst_height / nwindows)
        window_half_width = 40
        XCenterLeftWindow = indWhiteColumnL
        XCenterRightWindow = indWhiteColumnR
        left_lane_inds = np.array([], dtype=np.int16)
        right_lane_inds = np.array([], dtype=np.int16)

        #пустое изображение
        empt = np.zeros([dst_height, dst_width, 1], dtype=np.uint8)
        out_img = np.dstack((warped_yellow, empt, warped_white))

        #выделяем НЕчерные пиксели из белой и желтой линии
        nonzero_yellow = warped_yellow.nonzero()  # все ненулевые индексы массива по строкам и столбцам
        WhitePixelIndY_y = np.array(nonzero_yellow[0])  # y координаты всех белых пикселей (желтая линия)
        WhitePixelIndX_y = np.array(nonzero_yellow[1])  # x координаты всех белых пикселей (желтая линия)
        nonzero_white = warped_white.nonzero()
        WhitePixelIndY_w = np.array(nonzero_white[0])  # y координаты всех белых пикселей (белая линяя)
        WhitePixelIndX_w = np.array(nonzero_white[1])  # x координаты всех белых пикселей (белая линяя)

        for window in range(nwindows):
            win_y1 = out_img.shape[0] - (window + 1) * window_height
            win_y2 = out_img.shape[0] - (window) * window_height

            left_win_x1 = XCenterLeftWindow - window_half_width
            left_win_x2 = XCenterLeftWindow + window_half_width
            right_win_x1 = XCenterRightWindow - window_half_width
            right_win_x2 = XCenterRightWindow + window_half_width

            cv2.rectangle(out_img, (left_win_x1, win_y1), (left_win_x2, win_y2), (50 + window * 21, 0, 0), 2)
            cv2.rectangle(out_img, (right_win_x1, win_y1), (right_win_x2, win_y2), (0, 0, 50 + window * 21), 2)

            # выясняем какие белые пиксель попали внутрь окна
            # & - побитовое сложение, т.к. and не работает для массивов
            good_left_inds = ((WhitePixelIndY_y >= win_y1) & (WhitePixelIndY_y <= win_y2) & (WhitePixelIndX_y >= left_win_x1) & (
                    WhitePixelIndX_y <= left_win_x2)).nonzero()
            good_right_inds = ((WhitePixelIndY_w >= win_y1) & (WhitePixelIndY_w <= win_y2) & (WhitePixelIndX_w >= right_win_x1) & (
                    WhitePixelIndX_w <= right_win_x2)).nonzero()

            left_lane_inds = np.concatenate((left_lane_inds, good_left_inds))
            right_lane_inds = np.concatenate((right_lane_inds, good_right_inds))
            if len(good_left_inds) > 50:
                XCenterLeftWindow = np.int(np.mean(WhitePixelIndX_y[good_left_inds]))
            if len(good_right_inds) > 50:
                XCenterRightWindow = np.int(np.mean(WhitePixelIndX_w[good_right_inds]))

        # закрашиваем все белые пиксели внутри прямоугольников
        out_img[WhitePixelIndY_y[left_lane_inds], WhitePixelIndX_y[left_lane_inds]] = [255, 255, 255]
        out_img[WhitePixelIndY_w[right_lane_inds], WhitePixelIndX_w[right_lane_inds]] = [255, 255, 255]

        leftx = WhitePixelIndX_y[left_lane_inds]
        lefty = WhitePixelIndY_y[left_lane_inds]
        rightx = WhitePixelIndX_w[right_lane_inds]
        righty = WhitePixelIndY_w[right_lane_inds]

        if lefty.size and leftx.size and rightx.size and righty.size:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            center_fit = ((left_fit + right_fit) / 2)

            for ver_ind in range(out_img.shape[0]):
                gor_ind = (center_fit[0]) * (ver_ind ** 2) + (center_fit[1] * ver_ind) + center_fit[2]

                cv2.circle(out_img, (int(gor_ind), int(ver_ind)), 2, (255, 0, 255), 1)

            top_diff = 0
            mid_diff = 0
            bot_diff = 0


        cv2.imshow("frame", frame)
        cv2.imshow("yellow", warped_yellow_visual)
        cv2.imshow("white", warped_white_visual)
        cv2.imshow("out_img", out_img)


    else:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('screen.png', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()