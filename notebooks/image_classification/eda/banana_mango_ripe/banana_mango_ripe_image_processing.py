import os
import cv2
import numpy as np
from rembg import remove

input_folder = "../../../../datasets/banana_mango_ripe/train/images"
bg_removed_folder = "../../../../datasets/banana_mango_ripe/images_background_removed"
bg_removed_cropped_folder = "../../../../datasets/banana_mango_ripe/images_background_removed_cropped"

os.makedirs(bg_removed_folder, exist_ok=True)
os.makedirs(bg_removed_cropped_folder, exist_ok=True)

def process_image(image_path, filename):

    with open(image_path, "rb") as f:
        img_no_bg = remove(f.read())  

    img = cv2.imdecode(np.frombuffer(img_no_bg, np.uint8), cv2.IMREAD_UNCHANGED)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    bg_removed_path = os.path.join(bg_removed_folder, filename.replace(".jpg", "_no_bg.jpg"))
    cv2.imwrite(bg_removed_path, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img = img[y:y+h, x:x+w]  

    bg_removed_cropped_path = os.path.join(bg_removed_cropped_folder, filename.replace(".jpg", "_no_bg_cropped.jpg"))
    cv2.imwrite(bg_removed_cropped_path, img)

    return True

count = 0
for file in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file)

    if process_image(input_path, file):
        count += 1
        print(f"Processed {count}: {file}")

print("\nAll images have been processed and saved!")