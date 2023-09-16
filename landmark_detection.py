from google.cloud import vision_v1
from google.cloud import storage
import cv2
import numpy as np
import math

# Initialize the Google Cloud client for Vision.
client_vision = vision_v1.ImageAnnotatorClient()

# Replace with your GCP project ID, bucket name, and image file name.
project_id = "YOUR_PROJECT_ID"
bucket_name = "YOUR_BUCKET_NAME"
image_file_name = "YOUR_IMAGE_FILE"

# Define the GCS URI for the image
gcs_uri = f"gs://{bucket_name}/{image_file_name}"

# Perform landmark detection using Cloud Vision API.
image = vision_v1.Image()
image.source.image_uri = gcs_uri

response = client_vision.landmark_detection(
    image=image,
)

# Download the image from GCS.
storage_client = storage.Client(project=project_id)
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(image_file_name)
image_data = blob.download_as_bytes()

# Load the image using OpenCV.
image_cv = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
height, width, _ = image_cv.shape

FONT_SCALE = 2e-3 # Adjust for larger font size in all images.
THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images.

# Define text-related parameters.
font = cv2.FONT_HERSHEY_SIMPLEX
font_color = (0, 255, 0)  # White color in BGR format.
font_scale = min(width, height) * FONT_SCALE
line_thickness = math.ceil(min(width, height) * THICKNESS_SCALE)

# Initialize vertical position for writing text.
text_y = 200

# Write landmark names on the image if confidence is more than 0.5.
for landmark in response.landmark_annotations:
    # Extract landmark name and confidence score.
    landmark_name = landmark.description
    confidence = landmark.score

    # Check if confidence is more than 0.5.
    if confidence > 0.5:
        # Define the y-coordinate position to write the text.
        text_x = 100

        # Put the landmark name on the image.
        cv2.putText(
            image_cv,
            landmark_name,
            (text_x, text_y),
            font,
            font_scale,
            font_color,
            line_thickness,
            lineType=cv2.LINE_AA,
        )

        # Increment the vertical position for the next text.
        text_y += 50  # You can adjust this value to control spacing.

# Save the image with landmark names.
cv2.imwrite("landmark_detect_output.jpg", image_cv)
