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

