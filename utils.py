import cv2
import numpy as np

def preprocess_full_res(img_orig, width_small=640, height_small=640):
    height_orig, width_orig = img_orig.shape[:2]
    scale_x = width_orig / width_small
    scale_y = height_orig / height_small

    img_small = cv2.resize(img_orig, (width_small, height_small))
    img = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.Canny(img, 100, 250)
    kernel = np.ones((5, 5))
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=1)
    img_canny = img.copy()

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    biggest = np.array([])
    max_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5000:
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    img_contours = img_small.copy()
    if biggest.size != 0:
        cv2.drawContours(img_contours, [biggest], -1, (0, 255, 0), 3)

    if biggest.size == 0:
        return None, img_contours, img_canny

    pts = biggest.reshape((4, 2))
    new_pts = np.zeros((4, 1, 2), dtype=np.float32)

    add = pts.sum(axis=1)
    new_pts[0] = pts[np.argmin(add)]
    new_pts[3] = pts[np.argmax(add)]   

    diff = np.diff(pts, axis=1)
    new_pts[1] = pts[np.argmin(diff)]
    new_pts[2] = pts[np.argmax(diff)]

    new_pts_full = new_pts.copy()
    new_pts_full[:, 0, 0] *= scale_x
    new_pts_full[:, 0, 1] *= scale_y

    dest_width = int(width_small * scale_x)
    dest_height = int(height_small * scale_y)

    points2 = np.float32([
        [0, 0],
        [dest_width, 0],
        [0, dest_height],
        [dest_width, dest_height]
    ])

    matrix = cv2.getPerspectiveTransform(new_pts_full.astype(np.float32), points2)
    img_warped = cv2.warpPerspective(img_orig, matrix, (dest_width, dest_height))

    return img_warped, img_contours, img_canny