import torch

from transformers import AutoImageProcessor, AutoModelForObjectDetection
#from transformers import pipeline

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import io
from random import choice


image_processor_tiny = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
model_tiny = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

image_processor_small = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
model_small = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small")


import gradio as gr


COLORS = ["#ff7f7f", "#ff7fbf", "#ff7fff", "#bf7fff",
            "#7f7fff", "#7fbfff", "#7fffff", "#7fffbf",
            "#7fff7f", "#bfff7f", "#ffff7f", "#ffbf7f"]

fdic = {
    "family" : "DejaVu Serif",
    "style" : "normal",
    "size" : 18,
    "color" : "yellow",
    "weight" : "bold"
}


def get_figure(in_pil_img, in_results):
    plt.figure(figsize=(16, 10))
    plt.imshow(in_pil_img)
    ax = plt.gca()

    for score, label, box in zip(in_results["scores"], in_results["labels"], in_results["boxes"]):
        selected_color = choice(COLORS)

        box_int = [i.item() for i in torch.round(box).to(torch.int32)]
        x, y, w, h = box_int[0], box_int[1], box_int[2]-box_int[0], box_int[3]-box_int[1]
        #x, y, w, h = torch.round(box[0]).item(), torch.round(box[1]).item(), torch.round(box[2]-box[0]).item(), torch.round(box[3]-box[1]).item()

        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color=selected_color, linewidth=3, alpha=0.8))
        ax.text(x, y, f"{model_tiny.config.id2label[label.item()]}: {round(score.item()*100, 2)}%", fontdict=fdic, alpha=0.8)

    plt.axis("off")

    return plt.gcf()


def infer(in_pil_img, in_model="yolos-tiny", in_threshold=0.9):
    target_sizes = torch.tensor([in_pil_img.size[::-1]])

    if in_model == "yolos-small":
        inputs = image_processor_small(images=in_pil_img, return_tensors="pt")
        outputs = model_small(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        results = image_processor_small.post_process_object_detection(outputs, threshold=in_threshold, target_sizes=target_sizes)[0]

    else:
        inputs = image_processor_tiny(images=in_pil_img, return_tensors="pt")
        outputs = model_tiny(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        results = image_processor_tiny.post_process_object_detection(outputs, threshold=in_threshold, target_sizes=target_sizes)[0]

    figure = get_figure(in_pil_img, results)

    buf = io.BytesIO()
    figure.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    output_pil_img = Image.open(buf)

    return output_pil_img


with gr.Blocks(title="YOLOS Object Detection - ClassCat",
                    css=".gradio-container {background:lightyellow;}"
               ) as demo:
    #sample_index = gr.State([])

    gr.HTML("""<div style="font-family:'Times New Roman', 'Serif'; font-size:16pt; font-weight:bold; text-align:center; color:royalblue;">YOLOS Object Detection</div>""")

    gr.HTML("""<h4 style="color:navy;">1. Select a model.</h4>""")

    model = gr.Radio(["yolos-tiny", "yolos-small"], value="yolos-tiny", label="Model name")

    gr.HTML("""<br/>""")
    gr.HTML("""<h4 style="color:navy;">2-a. Select an example by clicking a thumbnail below.</h4>""")
    gr.HTML("""<h4 style="color:navy;">2-b. Or upload an image by clicking on the canvas.</h4>""")

    with gr.Row():
        input_image = gr.Image(label="Input image", type="pil")
        output_image = gr.Image(label="Output image with predicted instances", type="pil")

    gr.Examples(['samples/cats.jpg', 'samples/detectron2.png', 'samples/cat.jpg', 'samples/hotdog.jpg'], inputs=input_image)

    gr.HTML("""<br/>""")
    gr.HTML("""<h4 style="color:navy;">3. Set a threshold value (default to 0.9)</h4>""")

    threshold = gr.Slider(0, 1.0, value=0.9, label='threshold')

    gr.HTML("""<br/>""")
    gr.HTML("""<h4 style="color:navy;">4. Then, click "Infer" button to predict object instances. It will take about 10 seconds (yolos-tiny) or 20 seconds (yolos-small).</h4>""")

    send_btn = gr.Button("Infer")
    send_btn.click(fn=infer, inputs=[input_image, model, threshold], outputs=[output_image])

    gr.HTML("""<br/>""")
    gr.HTML("""<h4 style="color:navy;">Reference</h4>""")
    gr.HTML("""<ul>""")
    gr.HTML("""<li><a href="https://huggingface.co/docs/transformers/model_doc/yolos" target="_blank">Hugging Face Transformers - YOLOS</a>""")
    gr.HTML("""</ul>""")


#demo.queue()
demo.launch(share=True)




### EOF ###