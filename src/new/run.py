#!/usr/bin/env python3


from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub

#Load model.
model = hub.load('https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1')
model = model.signatures['default']

##Load image.
img = tf.io.read_file("/mnt/media/storage3/PICARCHIVE/2023-Q3/2023-07/text/2023-07-30_20-20-17-012.png")
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

#Run model.
outputs = model(img)
outputs = {k: v.numpy() for k, v in outputs.items()}
print(outputs)

#Draw boxes on image. (list of box)
def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", 25)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            _draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image