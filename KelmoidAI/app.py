import gradio as gr
from KelmoidAI_generator import OpenSCADGenerator
import os
import re
import json
from pathlib import Path
import subprocess
import tempfile 

# Initialize OpenSCAD generator
generator = OpenSCADGenerator()

# Path to history.json
temp_dir = Path(tempfile.mkdtemp())
history_file = temp_dir / "history.json"
if not history_file.exists():
    history_file.write_text("{}")  # Create an empty json file if not exists

def load_history():
    if history_file.exists():
        with open(history_file, "r") as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

def launch_openscad(filepath):
    """Modified for Hugging Face - can't launch OpenSCAD directly"""
    return "OpenSCAD file generated! Download and open in OpenSCAD to view."

def highlight_scad(code):
    """Simple OpenSCAD syntax highlighting"""
    if not code:
        return code

    keywords = ["module", "function", "include", "use", "if", "else", "for", "let",
                "cube", "sphere", "cylinder", "polyhedron", "translate", "rotate",
                "scale", "mirror", "union", "difference", "intersection",
                "$fn", "$fa", "$fs"]

    for kw in keywords:
        code = code.replace(kw, f'<span style="color: #0000FF;">{kw}</span>')

    code = re.sub(r'(//.*?$|/\*.*?\*/)',
                 r'<span style="color: #228B22;">\1</span>',
                 code, flags=re.DOTALL)

    return f'<pre style="font-family: monospace; background: #f5f5f5; padding: 10px;">{code}</pre>'

def generate_openscad(prompt):
    history = load_history()

    if prompt in history:
        code = history[prompt]
        message = "Loaded from previous correct generation ✅"
        filename = generator.generate_filename(prompt)
        generator.save_scad_file(code, filename)
        generator.last_generated_code = code  # Set the attribute
        launch_status = launch_openscad(filename)
        return highlight_scad(code), f"{message} | {launch_status}", filename

    code, message = generator.generate(prompt)
    if code:
        generator.last_generated_code = code
        filename = generator.generate_filename(prompt)
        generator.save_scad_file(code, filename)
        launch_status = launch_openscad(filename)  # Get the status
        return highlight_scad(code), launch_status, filename  # Use the status directly

    return "", "Error: No valid OpenSCAD code generated", None

def handle_feedback(prompt, feedback):
    if feedback == "Yes":
        history = load_history()
        history[prompt] = generator.last_generated_code  # Save only if feedback is Yes
        save_history(history)
    return gr.update(), gr.update(), gr.update()


def open_output_folder():
    return "In Hugging Face Spaces, files are downloaded directly rather than saved to a folder."



# Get absolute path to background image
current_dir = Path(__file__).parent
background_path = current_dir / "background.jpg"  # Change to your image file if needed

# CSS for background and panel styling
custom_css = f"""
:root {{
    --panel-bg: rgba(255, 255, 255, 0.85);
    --panel-border-radius: 10px;
}}
body {{
    background-image: url('{background_path.as_posix()}');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    min-height: 100vh;
    margin: 0;
}}
.gradio-container {{
    background: transparent !important;
    max-width: 90% !important;
    margin: 0 auto !important;
    padding: 20px !important;
}}
.gradio-interface {{
    background: var(--panel-bg) !important;
    border-radius: var(--panel-border-radius) !important;
    padding: 20px !important;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(0, 0, 0, 0.1);
}}
.gradio-interface .panel {{
    background: var(--panel-bg) !important;
    border-radius: var(--panel-border-radius) !important;
}}

"""
custom_css += """
#centered-feedback {
    display: flex;
    flex-direction: column; /* Stack label and options vertically */
    align-items: center;    /* Center align everything */
    gap: 10px;              /* Add spacing between label and options */
}
#centered-feedback label {
    text-align: center;     /* Center the label text */
    margin-bottom: 5px;     /* Add spacing below the label */
}
"""

custom_css += """
#feedback-btn {
    color: white;             /* White text */
    font-size: 16px;          /* Larger font size */
    padding: 10px 20px;       /* Add padding */
    border-radius: 8px;       /* Rounded corners */
    border: none;             /* Remove border */
    cursor: pointer;          /* Pointer cursor on hover */
    transition: background-color 0.3s ease; /* Smooth hover effect */
}
"""
custom_css += """
#centered-feedback {
    display: flex;
    justify-content: center; /* Center horizontally */
    align-items: center;    /* Center vertically */
    gap: 10px;              /* Add spacing between options */
}
#centered-feedback label {
    margin: 0;              /* Remove default margin for labels */
}"""

# Gradio App
with gr.Blocks(title="KelmoidAI Genesis LLM", theme=gr.themes.Soft(), css=custom_css) as app:
    gr.Markdown("# 🏗️ KelmoidAI Genesis LLM")

    with gr.Column(variant="panel"):
        with gr.Row():
            prompt = gr.Textbox(
                label="Describe your 3D model",
                lines=3,
                placeholder="Example: 'A 50mm diameter gear with 12 teeth and 5mm thickness'",
                container=False
            )
        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary", scale=1)

        status = gr.Textbox(label="Status", interactive=False, visible=True)
        code_display = gr.HTML(label="Generated OpenSCAD Code")

        
        feedback_col = gr.Column()
        with feedback_col:
            with gr.Row():
                feedback = gr.Radio(
                    ["Yes", "No"], 
                    label="Did the generated code give the desired output?",
                    elem_id="centered-feedback"
                )
            with gr.Row():
                submit_feedback_btn = gr.Button(
                    "✅ Submit Feedback", 
                    variant="primary",  # Makes it stand out
                    scale=2,            # Increases the size
                    elem_id="feedback-btn"  # Add a custom ID for further CSS styling
                )

        with gr.Row():
            open_folder_btn = gr.Button("Open Outputs Folder")
            open_openscad_btn = gr.Button("Open in OpenSCAD", visible=False)
            download_btn = gr.DownloadButton(
                "Download SCAD File",
                visible=False
            )

    # Save filename and last generation state
    current_file = gr.State()

    # Event handlers
    generate_btn.click(
        fn=generate_openscad,
        inputs=[prompt],
        outputs=[code_display, status, current_file]
    ).then(
        lambda x: gr.DownloadButton(visible=bool(x)), 
        inputs=[current_file],
        outputs=[download_btn]
    ).then(
        lambda x: gr.Button(visible=bool(x)),  # Show "Open in OpenSCAD" if file exists
        inputs=[current_file],
        outputs=[open_openscad_btn]
    )

    submit_feedback_btn.click(
        fn=handle_feedback,
        inputs=[prompt, feedback],
        outputs=[code_display, status, current_file]
    )

    download_btn.click(
        lambda x: x if os.path.exists(x) else None,
        inputs=[current_file],
        outputs=[download_btn]
    )

    open_openscad_btn.click(
        fn=lambda f: launch_openscad(f) if f else "No file to open",
        inputs=[current_file],
        outputs=[status]
    )

    open_folder_btn.click(
        fn=open_output_folder,
        outputs=[status]
    ).then(
        inputs=[status],
        outputs=[feedback_col]
    )

if __name__ == "__main__":
    if not background_path.exists():
        print(f"Warning: Background image not found at {background_path}")
        print("Using default background instead")
    
    # Create outputs directory if it doesn't exist
    Path("outputs").mkdir(exist_ok=True)
    app.launch()