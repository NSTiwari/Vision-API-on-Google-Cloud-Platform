from typing import Sequence
from google.cloud import vision
from google.cloud import storage
from PIL import Image, ImageDraw
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
image_file_name = "YOUR_IMAGE_FILE"

# Define the GCS URI for the image.
gcs_uri = f"gs://{bucket_name}//{image_file_name}"

# Define the feature.
features = [vision.Feature.Type.FACE_DETECTION]

# Perform face detection using Cloud Vision API.
response = analyze_image_from_uri(gcs_uri, features)
print(response)

# Download the image from GCS.
storage_client = storage.Client(project=project_id)
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(image_file_name)
image_data = blob.download_as_bytes()

# Create a PIL representation of the image to draw the bounding boxes.
image = Image.open(BytesIO(image_data))
draw = ImageDraw.Draw(image)

for face in response.face_annotations:
    # Convert the bounding box vertices to the format expected by PIL.
    box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
    # Draw a rectangle around the face.
    draw.polygon(box, outline=(0, 255, 0), width=2)

# Save the image with the bounding boxes.
output_image_file_name = "face_detection_output.jpg"
image.save(output_image_file_name)
print(f"Saved image with bounding boxes to {output_image_file_name}.")
