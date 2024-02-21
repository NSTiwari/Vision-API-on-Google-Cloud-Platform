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
features = [vision.Feature.Type.LANDMARK_DETECTION]

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
font = ImageFont.load_default(size=100)

for landmark in response.landmark_annotations:
    # Convert the bounding box vertices to the format expected by PIL.
    box = [(vertex.x, vertex.y) for vertex in landmark.bounding_poly.vertices]
    # Draw a rectangle around the landmark.
    draw.polygon(box, outline=(0, 255, 0), width=2)
    # Get the landmark text and write it on the image.
    landmark_text = landmark.description
    draw.text((100, 100), landmark_text, font=font, fill=(255, 0, 0))

# Save the image with the bounding boxes and landmark text.
output_image_file_name = "landmark_detection_output.jpg"
image.save(output_image_file_name)
print(f"Saved image with landmark detection to {output_image_file_name}.")
