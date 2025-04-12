import gradio as gr

css = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

body {
    background: linear-gradient(to bottom, #79c152 0%, #79c152 60%, #5c432d 100%) !important;
    font-family: 'Press Start 2P', cursive !important;
    color: #ffffff;
}

.gr-button {
    background-color: #3e8527 !important;
    border: 2px solid #254d16 !important;
    color: #ffffff !important;
}

.gr-button:hover {
    background-color: #6fcf44 !important;
    border-color: #4a6c2d !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🌱 Minecraft 草地界面")
    gr.Textbox(label="你想说啥")
    gr.Button("点我")

demo.launch()
