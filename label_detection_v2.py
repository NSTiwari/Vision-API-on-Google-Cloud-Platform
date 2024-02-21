from typing import Sequence
from google.cloud import vision
from google.cloud import storage
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Initialize the Google Cloud client for Vision.
def analyze_image_from_uri(
    image_uri: str,
    feature_types: Sequence,
) -> vision.AnnotateImageResponse:
    client = vision.ImageAnnotatorClient()

    image = vision.Image()
    image.source.image_uri = image_uri
    features = [vision.Feature(type_=feature_type) for feature_type in feature_types]
    request = vision.AnnotateImageRequest(image=image, features=features)

    response = client.annotate_image(request=request)

    return response

# Replace with your GCP project ID, bucket name, and image file name.
project_id = "YOUR_PROJECT_ID"
bucket_name = "YOUR_BUCKET_NAME"
image_file_name = "YOUR_IMAGE_NAME"

# Define the GCS URI for the image.
gcs_uri = f"gs://{bucket_name}//{image_file_name}"

# Define the feature.
features = [vision.Feature.Type.LABEL_DETECTION]

# Perform face detection using Cloud Vision API.
response = analyze_image_from_uri(gcs_uri, features)
print(response)

# Download the image from GCS.
storage_client = storage.Client(project=project_id)
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(image_file_name)
image_data = blob.download_as_bytes()

# Create a PIL representation of the image to draw the bounding boxes and write the landmark text.
image = Image.open(BytesIO(image_data))
draw = ImageDraw.Draw(image)

# Load a font for the landmark text.
font = ImageFont.load_default(size=10)

# Define the top N detections.
NUM_DETECTIONS = 5

# Get the first label annotation (highest confidence).
for i in range(NUM_DETECTIONS):
    label_annotation = response.label_annotations[i]
    label_text = label_annotation.description
    # Write the lable text on the image.
    draw.text((10, 10 + 10*i), label_text, font=font, fill=(0, 255, 0))

# Save the image with the bounding box and label text.
output_image_file_name = "label_detection_output.jpg"
image.save(output_image_file_name)
print(f"Saved image with label text to {output_image_file_name}.")
