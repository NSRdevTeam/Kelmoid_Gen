#!/usr/bin/env python3
"""
Simple test for AdvancedCAD primitives
"""

import sys
import os
sys.path.append('src')

try:
    from core.primitives import CubeGenerator, SphereGenerator
    print("Successfully imported primitive generators")
    
    # Test cube generation
    cube_gen = CubeGenerator()
    cube_mesh = cube_gen.generate({'size': [10, 10, 10]})
    print(f"Generated cube: {cube_mesh.vertices.shape[0]} vertices, {cube_mesh.faces.shape[0]} faces")
    
    # Test sphere generation  
    sphere_gen = SphereGenerator()
    sphere_mesh = sphere_gen.generate({'radius': 5, 'resolution': 16})
    print(f"Generated sphere: {sphere_mesh.vertices.shape[0]} vertices, {sphere_mesh.faces.shape[0]} faces")
    
    print("Core primitives working!")
    
except Exception as e:
    print(f"Error testing primitives: {e}")
    import traceback
    traceback.print_exc()
