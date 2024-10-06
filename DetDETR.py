import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image, ImageDraw
import requests


# Load the pre-trained DETR model and processor
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", force_download=True)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", force_download=True)


# Load an image (can be from a local path or URL)
def load_image(image_path_or_url):
    if image_path_or_url.startswith('http'):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw)
    else:
        image = Image.open(image_path_or_url)
    return image

# Perform object detection
def detect_objects(image):
    # Preprocess the image and prepare it for the model
    inputs = processor(images=image, return_tensors="pt")
    
    # Perform inference
    outputs = model(**inputs)

    # Convert logits to bounding boxes and class labels
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
    
    # Unpack results
    boxes, scores, labels = results['boxes'], results['scores'], results['labels']
    
    return boxes, scores, labels

# Visualize the detections on the image
def visualize_detections(image, boxes, scores, labels, score_threshold=0.7):
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        if score >= score_threshold:
            box = [round(i, 2) for i in box.tolist()]
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {score:.2f}", fill="red")
    return image

# Main function to load an image, detect objects, and display the result
# Main function to load an image, detect objects, and save/display the result
def main(image_path_or_url):
    image = load_image(image_path_or_url)
    boxes, scores, labels = detect_objects(image)
    image_with_detections = visualize_detections(image, boxes, scores, labels)
    
    # Save the image with detections to disk
    output_path = image_path_or_url.replace(".png", "_detections.png")  # Modify the output filename
    image_with_detections.save(output_path)
    print(f"Image with detections saved to {output_path}")


# Example usage:
image_path = ""  # Path to image
main(image_path)


