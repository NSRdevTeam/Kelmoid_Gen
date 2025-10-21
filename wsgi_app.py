import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr

# Try importing your Gradio interface
try:
    import app
except Exception as e:
    raise RuntimeError(f"Failed to import app.py: {e}")

# Detect the Gradio demo object
demo = None
for var_name in dir(app):
    obj = getattr(app, var_name)
    if isinstance(obj, gr.Blocks) or isinstance(obj, gr.Interface):
        demo = obj
        break

if demo is None:
    raise RuntimeError("No Gradio Blocks/Interface found in app.py — make sure your app defines one (e.g. demo = gr.Blocks(...))")

# Wrap Gradio app into FastAPI, then WSGI
from fastapi import FastAPI
from a2wsgi import ASGIMiddleware

fastapi_app = FastAPI()
gr.mount_gradio_app(fastapi_app, demo, path="/")

# Passenger expects a WSGI callable named 'application'
application = ASGIMiddleware(fastapi_app)
