import cv2
from skimage.metrics import structural_similarity as ssim


def imdiff(im1, im2) -> list:
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    _, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")

    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def main():
    img1 = cv2.imread("./assets/1.png")
    img2 = cv2.imread("./assets/2.png")

    contours = imdiff(img1, img2)

    img1_overlay = img1.copy()
    img2_overlay = img2.copy()

    for contour in contours:
        # if cv2.contourArea(contour) > 100:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), cv2.FILLED)
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), cv2.FILLED)

    alpha = 0.8

    img1 = cv2.addWeighted(img1_overlay, alpha, img1, 1 - alpha, 0)
    img2 = cv2.addWeighted(img2_overlay, alpha, img2, 1 - alpha, 0)

    # cv2.imshow("Image 1", img1)
    # cv2.imshow("Image 2", img2)

    # cv2.waitKey()
    # cv2.destroyAllWindows()

    cv2.imwrite("./assets/1_copy.png", cv2.resize(img1, None, fx=0.5, fy=0.5))
    cv2.imwrite("./assets/2_copy.png", cv2.resize(img2, None, fx=0.5, fy=0.5))


if __name__ == "__main__":
    main()
