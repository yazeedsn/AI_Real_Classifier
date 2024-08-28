import os 
import torch
import gradio as gr

import config
from torch.nn.functional import softmax
if os.path.isfile(config.WEIGTHS_PATH):
    model_state_dict = torch.load(config.WEIGTHS_PATH, weights_only=True)
    config.MODEL.load_state_dict(model_state_dict)

def predict(X):
   config.MODEL.eval()
   with torch.inference_mode():
    X = config.INPUT_TRANSFORM(X)
    y = config.MODEL(X)
    y = softmax(y, dim=1)
    i = y.argmax(dim=1).item()
    return config.IDX_TO_CLASS[i]


def detect(image):
    y = predict(image)
    return y

demo = gr.Interface(fn=detect, inputs=[gr.Image(sources='upload',type='numpy')], outputs=['text'])
demo.launch()