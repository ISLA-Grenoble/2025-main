import cv2
import numpy as np
import os
from glob import glob
from PIL import Image


def auto_crop_black_background(image_path, output_path, threshold=10):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a mask where the black background is detected
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours of the non-black regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(np.vstack(contours))

        # Crop the image
        cropped_image = image[y:y+h, x:x+w]

        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)

# Process all images in a folder
input_folder = "photos_original"
output_folder = "photos_crop"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        auto_crop_black_background(input_path, output_path)

print("Cropping completed!")

# Load OpenCV's pre-trained Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop_face(image_path, output_path):
    # Read image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y+h, x:x+w]  # Crop the face

            # Save each detected face (use indexing if multiple faces exist)
            face_output_path = output_path.replace(".jpg", f"_face_{i}.jpg")
            cv2.imwrite(face_output_path, face)

# Process all images in the processed folder
input_folder = "photos_crop"
output_folder = "photos_face"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        detect_and_crop_face(input_path, output_path)

print("Face extraction completed!")

list_photos_face = glob('photos_face/*')
for photo_face in list_photos_face:
    photo_face_new = photo_face.split('/')[1].split('_')[0] + '.jpg'
    os.rename(photo_face, 'photos_face/' + photo_face_new)

list_photos_face = glob('photos_face/*')
for photo_face in list_photos_face:
    base_width = 128
    img = Image.open(photo_face)
    img = img.resize((128, 128), Image.Resampling.LANCZOS)
    img.save(photo_face)