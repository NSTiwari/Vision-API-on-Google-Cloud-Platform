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

# Perform text detection using Cloud Vision API.
image = vision_v1.Image()
image.source.image_uri = gcs_uri

response = client_vision.text_detection(
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
font_scale = 0.8
font_color = (0, 0, 255)  # White color in BGR format.

# Draw bounding boxes around detected text and write the text.
for page in response.full_text_annotation.pages:
    for block in page.blocks:
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                vertices = [word.bounding_box.vertices[0].x, word.bounding_box.vertices[0].y,
                            word.bounding_box.vertices[2].x, word.bounding_box.vertices[2].y]

                # Convert vertices to integers.
                vertices = [int(v) for v in vertices]

                # Draw a rectangle around the text.
                cv2.rectangle(image_cv, (vertices[0], vertices[1]), (vertices[2], vertices[3]), box_color, line_thickness)

                # Extract and write the text.
                text = " ".join([symbol.text for symbol in word.symbols])
                text_x = vertices[0]
                text_y = vertices[1] - 10  # Adjust the vertical position of the text.

                # Put the text on the image.
                cv2.putText(
                    image_cv,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    font_color,
                    line_thickness,
                    lineType=cv2.LINE_AA,
                )

# Save the image with bounding boxes and text.
cv2.imwrite("text_detect_output.jpg", image_cv)