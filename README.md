---
title: Kelmoid Genesis LLM Prototype
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.47.2
app_file: app.py
pinned: false
license: mit
---

# Kelmoid Genesis LLM Prototype

A comprehensive CAD engineering suite powered by AI and natural language processing.

## Features

- 🎨 **Text-to-CAD Generator**: Create 3D models from natural language descriptions
- 📐 **2D Plate Designer**: Generate technical drawings and G-code for manufacturing  
- 🌊 **CFD Simulator**: Computational fluid dynamics analysis
- 📋 **Orthographic Views**: Generate technical drawings from 3D models

## How to Use

1. **Text-to-CAD Generator**: Describe what you want to create (e.g., "Create a steel cube 20x20x20mm")
2. **2D Plate Designer**: Describe plates with dimensions and features (e.g., "100mm x 50mm plate with 20mm hole")
3. **CFD Simulator**: Set up fluid flow parameters and run simulations

Simply use natural language to describe what you want to create, and the system will generate the corresponding CAD models, technical drawings, or perform fluid dynamics simulations.

## Technology Stack

- **Frontend**: Gradio
- **3D Modeling**: Trimesh, PyVista
- **Visualization**: Plotly, Matplotlib
- **CAD Processing**: OpenCV, Shapely
- **CFD**: NumPy-based solver

## License

MIT License
