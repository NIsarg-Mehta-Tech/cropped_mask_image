import cv2 as cv

org_img = cv.imread('images/original_image.png')
mask_img = cv.imread('images/masked_image.png', cv.IMREAD_GRAYSCALE)

masked = cv.bitwise_and(org_img,org_img, mask=mask_img)
resize_mask = cv.resize(masked, (500,500))

contours, _ = cv.findContours(mask_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

cropped_objs = []

for i, contour in enumerate(contours):
    x, y, w, h = cv.boundingRect(contour)
    cropped_object = org_img[y:y+h, x:x+w]

    cropped_resized = cv.resize(cropped_object, (200, 200))
    cropped_objs.append(cropped_resized)

    cv.imshow(f"Cropped Object {i+1}", cropped_resized)

cv.imshow("Mask Applied to Image", resize_mask)

cv.waitKey(0)
cv.destroyAllWindows()
