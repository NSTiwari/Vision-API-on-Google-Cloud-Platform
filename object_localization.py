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

# Perform object localization using Cloud Vision API.
image = vision_v1.Image()
image.source.image_uri = gcs_uri

response = client_vision.object_localization(
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

# Define bounding box color and line thickness.
box_color = (0, 255, 0)  # Green color in BGR format.
line_thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_color = (0, 0, 255)  # White color in BGR format.

# Draw bounding boxes and labels on the image.
for localized_object in response.localized_object_annotations:
    vertices = localized_object.bounding_poly.normalized_vertices
    points = [(int(vertex.x * image_cv.shape[1]), int(vertex.y * image_cv.shape[0])) for vertex in vertices]

    # Draw a rectangle around the object.
    cv2.rectangle(image_cv, points[0], points[2], box_color, line_thickness)

    # Get the label of the object.
    label = localized_object.name

    # Calculate the position to write the label.
    label_x = points[0][0]
    label_y = points[0][1] - 10  # Adjust the vertical position of the label.

    # Write the label on the image.
    cv2.putText(
        image_cv,
        label,
        (label_x, label_y),
        font,
        font_scale,
        font_color,
        line_thickness,
        lineType=cv2.LINE_AA,
    )

# Save the image with bounding boxes and labels.
cv2.imwrite("object_localize_output.jpg", image_cv)