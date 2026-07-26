import cv2
import numpy as np

# ---------------------- CONFIG ----------------------
IMAGE_PATH = "./input/1.jpg"
WIDTH, HEIGHT = 640, 480
# -----------------------------------------------------


def preprocess(img):
    """Convert to gray, blur, edge-detect, dilate, erode."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blur, 60, 60)
    kernel = np.ones((5, 5))
    dilated = cv2.dilate(canny, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded


def reorder_points(pts):
    """Reorder 4 corner points to: top-left, top-right, bottom-left, bottom-right."""
    pts = pts.reshape((4, 2))
    new_pts = np.zeros((4, 1, 2), dtype=np.float32)

    add = pts.sum(axis=1)
    new_pts[0] = pts[np.argmin(add)]   # top-left has smallest sum
    new_pts[3] = pts[np.argmax(add)]   # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    new_pts[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    new_pts[2] = pts[np.argmax(diff)]  # bottom-left has largest difference

    return new_pts


def get_biggest_contour(img):
    """Find the largest 4-point contour (assumed to be the document)."""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = np.array([])
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area


def draw_contours(img, biggest):
    """Draw the detected document contour and corner points on a copy of the image."""
    out = img.copy()
    if biggest.size != 0:
        cv2.drawContours(out, [biggest], -1, (0, 255, 0), 5)
        for point in biggest.reshape(4, 2):
            cv2.circle(out, tuple(point.astype(int)), 8, (0, 0, 255), cv2.FILLED)
    return out


def warp_image(img, biggest, w, h):
    """Apply perspective transform to get a top-down view of the document."""
    biggest = reorder_points(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (w, h))

    # crop slightly to remove border noise, then resize back
    warped = warped[10:warped.shape[0] - 10, 10:warped.shape[1] - 10]
    warped = cv2.resize(warped, (w, h))
    return warped


def stack_images(scale, img_array):
    """Stack a grid of images (any mix of gray/color) into one window."""
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1] if rows_available else img_array[0].shape[1]
    height = img_array[0][0].shape[0] if rows_available else img_array[0].shape[0]

    if rows_available:
        for x in range(rows):
            for y in range(cols):
                img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        blank = np.zeros((height, width, 3), np.uint8)
        hor = [blank] * rows
        for x in range(rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(img_array)
    return ver


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Could not read image at {IMAGE_PATH}")
        return

    img = cv2.resize(img, (WIDTH, HEIGHT))
    blank = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    processed = preprocess(img)
    biggest, max_area = get_biggest_contour(processed)
    contour_img = draw_contours(img, biggest)

    if biggest.size != 0:
        warped = warp_image(img, biggest, WIDTH, HEIGHT)
    else:
        warped = blank.copy()
        print("No 4-point document contour found. Try adjusting Canny thresholds or area filter.")

    stacked = stack_images(0.75, [[img, processed],
                                   [contour_img, warped]])

    cv2.imshow("Document Scanner - Stacked (Original | Processed | Contours | Warped)", stacked)
    cv2.imshow("Final Scanned Output", warped)

    print("Press 's' to save the scanned output, any other key to exit.")
    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        cv2.imwrite("scanned_output.jpg", warped)
        print("Saved scanned_output.jpg")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()