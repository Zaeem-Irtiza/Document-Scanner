import cv2
import numpy as np

width = 640
height = 640

img_orig = cv2.imread("./input/7.jpeg")
img_orig = cv2.resize(img_orig, (width, height))
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


temp = biggest.copy()
biggest[1][0] = temp[3][0]
biggest[2][0] = temp[1][0]
biggest[3][0] = temp[2][0]

points1 = np.float32(biggest)
points2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

matrix = cv2.getPerspectiveTransform(points1, points2)

img_warped = cv2.warpPerspective(img_final, matrix, (width, height))

cv2.imwrite("./output.jpeg", img_warped)

cv2.imshow("Original Image", img_orig)
cv2.imshow("Warped Image", img_warped)
cv2.waitKey(0)
cv2.destroyAllWindows()