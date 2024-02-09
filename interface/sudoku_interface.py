import gradio as gr
from helpers import predict_string

input_string = gr.components.Text(label="Input")
output_grid = gr.components.Dataframe(label="Output", row_count=9, col_count=9)
demo = gr.Interface(
    fn=predict_string,
    inputs=input_string,
    outputs=output_grid,
    allow_flagging="never",
    theme=gr.themes.Monochrome(),
    css="footer{display:none !important} table{overflow: hidden !important} thead{display: none !important}",
)
demo.queue().launch()
