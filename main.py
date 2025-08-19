import cv2 as cv

org_img = cv.imread('images/original_image.png')
mask_img = cv.imread('images/masked_image.png', cv.IMREAD_GRAYSCALE)

_, mask_img = cv.threshold(mask_img, 127, 255, cv.THRESH_BINARY)

final_image = cv.bitwise_and(org_img, org_img, mask=mask_img)

contours, _ = cv.findContours(mask_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cropped_objs = []

for i, contour in enumerate(contours):
    x, y, w, h = cv.boundingRect(contour)
    cropped_object = final_image[y:y+h, x:x+w]

    cropped_resized = cv.resize(cropped_object, (200, 200))
    cropped_objs.append(cropped_resized)

    cv.imshow(f"Cropped Object {i+1}", cropped_resized)

cv.waitKey(0)
cv.destroyAllWindows()
