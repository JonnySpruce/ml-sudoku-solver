import gradio as gr
from constants import *
from helpers import predict_string

input_string = gr.components.Text(label="Input")
model_choice = gr.components.Dropdown(
    [LINEAR, CONVOLUTIONAL], label="Model", value=LINEAR
)
immediate_or_best = gr.components.Radio(
    [IMMEDIATE, BEST], label="Prediction Type", value=IMMEDIATE
)
output_grid = gr.components.Dataframe(label="Output", row_count=9, col_count=9)

demo = gr.Interface(
    fn=predict_string,
    inputs=[input_string, model_choice, immediate_or_best],
    outputs=output_grid,
    allow_flagging="never",
    theme=gr.themes.Monochrome(),
    css="footer{display:none !important} table{overflow: hidden !important} thead{display: none !important}",
)
demo.queue().launch()
