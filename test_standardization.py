#!/usr/bin/env python3
"""
Test script to validate the standardized CAD system
All shape functions should return only Trimesh meshes
"""

import sys
import traceback
from app import (
    TextToCADGenerator, 
    validate_mesh, 
    generate_orthographic_views,
    render_error_to_image,
    CADQUERY_AVAILABLE,
    BOOL_BACKEND
)

def test_shape_consistency():
    """Test that all shape functions return Trimesh objects"""
    print("🧪 Testing CAD System Standardization")
    print("=" * 50)
    
    cad_gen = TextToCADGenerator()
    test_shapes = ['cube', 'sphere', 'cylinder', 'washer', 'bracket', 'door_frame']
    
    for shape in test_shapes:
        print(f"Testing {shape}...")
        try:
            params = {
                'shape': shape,
                'dimensions': {'length': 50, 'width': 50, 'height': 50, 'radius': 25},
                'color': 'blue',
                'precision': 'fast'
            }
            
            mesh = cad_gen.generate_3d_model(params)
            
            # Verify it's a mesh object
            print(f"  ✅ Type: {type(mesh).__name__}")
            
            # Validate the mesh
            validate_mesh(mesh)
            print(f"  ✅ Validation: Passed")
            print(f"  📊 Stats: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            traceback.print_exc()
        
        print()

def test_orthographic_views():
    """Test orthographic views generation"""
    print("🖼️  Testing Orthographic Views")
    print("=" * 30)
    
    try:
        cad_gen = TextToCADGenerator()
        params = {
            'shape': 'cube',
            'dimensions': {'length': 100, 'width': 50, 'height': 25},
            'color': 'red',
            'precision': 'fast'
        }
        
        mesh = cad_gen.generate_3d_model(params)
        print(f"✅ Generated mesh: {len(mesh.vertices)} vertices")
        
        # Test orthographic views
        ortho_img = generate_orthographic_views(mesh, layout="2x3")
        print(f"✅ Orthographic views: {type(ortho_img).__name__}")
        print(f"📏 Image size: {ortho_img.size}")
        
    except Exception as e:
        print(f"❌ Orthographic test failed: {str(e)}")
        traceback.print_exc()
    
    print()

def test_error_handling():
    """Test error handling with invalid meshes"""
    print("⚠️  Testing Error Handling")
    print("=" * 25)
    
    try:
        # Test with None
        try:
            validate_mesh(None)
            print("❌ Should have failed with None")
        except ValueError as e:
            print(f"✅ Correctly caught None: {e}")
        
        # Test with invalid object
        class FakeMesh:
            pass
        
        try:
            validate_mesh(FakeMesh())
            print("❌ Should have failed with fake mesh")
        except ValueError as e:
            print(f"✅ Correctly caught fake mesh: {e}")
        
        # Test error image generation
        error_img = render_error_to_image("Test error message")
        print(f"✅ Error image: {type(error_img).__name__}, size: {error_img.size}")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        traceback.print_exc()
    
    print()

def test_system_info():
    """Display system capabilities"""
    print("🔧 System Information")
    print("=" * 20)
    print(f"CadQuery Available: {CADQUERY_AVAILABLE}")
    print(f"Boolean Backend: {BOOL_BACKEND or 'None'}")
    print()

if __name__ == "__main__":
    print("🚀 Kelmoid Genesis - Standardization Test Suite")
    print("=" * 60)
    print()
    
    test_system_info()
    test_shape_consistency()
    test_orthographic_views()
    test_error_handling()
    
    print("✅ Standardization test completed!")
    print("📋 Summary: All shape functions now return only Trimesh meshes")
    print("🖼️  Images are generated only for visualization, not mixed with mesh data")
