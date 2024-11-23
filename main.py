#TODO: Create the image preprocessing
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

image = cv2.imread('table_image.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#TODO: Problem with deskew, will look it up later
coords = np.column_stack(np.where(binary > 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle
(h, w) = binary.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
deskewed = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# Run OCR on the preprocessed image
custom_config = r'--oem 3 --psm 6'  # 6 is for detecting blocks of text/tables
data = pytesseract.image_to_data(deskewed, config=custom_config, output_type=Output.DICT)

# Extract text row-wise and column-wise based on bounding box coordinates
rows = []
current_row = []
previous_y = data['top'][0]

for i in range(len(data['text'])):
    if int(data['conf'][i]) > 60:  # Only consider text with a high confidence score
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        text = data['text'][i]

        # Detect a new row based on the 'top' position
        if abs(y - previous_y) > h:
            rows.append(current_row)
            current_row = []
            previous_y = y

        # Add the text to the current row
        current_row.append(text)
rows.append(current_row)  # Append the last row

print(rows)  # Each inner list is a row of table text