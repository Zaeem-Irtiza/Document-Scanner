import cv2
import numpy as np

img_orig = cv2.imread("./input/7.jpeg")
height_orig, width_orig = img_orig.shape[:2]

height_small, width_small = 640, 640
scale_x = width_orig/width_small
scale_y = height_orig/height_small

img_orig = cv2.resize(img_orig, (width_small, height_small))
img_final = img_orig.copy()
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.Canny(img, 95, 300)
kernel = np.ones((5, 5))
img = cv2.dilate(img, kernel, iterations=2)
img = cv2.erode(img, kernel, iterations=1)
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
biggest = np.array([])
max_area = 0
for c in contours:
    area = cv2.contourArea(c)
    if area > 5000:
        peri = cv2.arcLength(c, True)
        # print(peri, "perimeter")
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print(approx, "approx")
        # print(f"  -> corners={len(approx)}")
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area
if biggest.size != 0:
    cv2.drawContours(img_orig, [biggest], -1, (255, 0, 0), 2)

pts = biggest.reshape((4, 2))
new_pts = np.zeros((4, 1, 2), dtype=np.float32)

add = pts.sum(axis=1)
new_pts[0] = pts[np.argmin(add)]
new_pts[3] = pts[np.argmax(add)]

diff = np.diff(pts, axis=1)
new_pts[1] = pts[np.argmin(diff)]
new_pts[2] = pts[np.argmax(diff)]

matrix = cv2.getPerspectiveTransform(new_pts.astype(np.float32), np.array([[0, 0], [width_small, 0], [0, height_small], [width_small, height_small]], dtype=np.float32))
img_warped = cv2.warpPerspective(img_final, matrix, (width_small, height_small))
cv2.imwrite("./output.jpeg", img_warped)

cv2.imshow("Original Image", img_orig)
cv2.imshow("Warped Image", img_warped)
cv2.waitKey(0)
cv2.destroyAllWindows()