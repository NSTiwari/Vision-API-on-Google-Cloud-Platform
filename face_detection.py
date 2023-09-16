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

# Define the GCS URI for the image.
gcs_uri = f"gs://{bucket_name}/input/{image_file_name}"

# Perform face detection using Cloud Vision API.
image = vision_v1.Image()
image.source.image_uri = gcs_uri

response = client_vision.face_detection(
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

# Define bounding box color and line thickness.
THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images.
line_thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
box_color = (0, 255, 0)  # Green color in BGR format.

# Draw bounding boxes around detected faces.
for face in response.face_annotations:
    vertices = face.bounding_poly.vertices
    points = [(vertex.x, vertex.y) for vertex in vertices]

    # Draw a rectangle around the face.
    cv2.rectangle(image_cv, points[0], points[2], box_color, line_thickness)

# Save the image with bounding boxes.
cv2.imwrite("face_detect_output.jpg", image_cv)