"""
Comprehensive test suite for AdvancedCAD core functionality
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config import AppConfig
from core.parser import parse_script, ScriptLexer, TokenType
from core.primitives import create_primitive, primitive_factory
from core.csg_operations import union, difference, intersection, hull, minkowski
from core.extrusions import create_2d_shape, linear_extrude, rotate_extrude
from core.transforms import (
    translate_mesh, rotate_mesh, scale_mesh, mirror_mesh,
    create_translation_matrix, create_rotation_matrix, create_scale_matrix
)
from core.file_io import export_mesh, import_mesh


class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def test_config_creation(self):
        """Test configuration creation"""
        config = AppConfig()
        self.assertIsNotNone(config)
        self.assertIsInstance(config.config, dict)
    
    def test_config_get_set(self):
        """Test configuration get/set operations"""
        config = AppConfig()
        
        # Test getting default value
        theme = config.get("ui.theme", "light")
        self.assertEqual(theme, "dark")  # Default is dark
        
        # Test setting value
        config.set("ui.theme", "light")
        theme = config.get("ui.theme")
        self.assertEqual(theme, "light")
    
    def test_recent_files(self):
        """Test recent files management"""
        config = AppConfig()
        
        # Add recent files
        config.add_recent_file("/test/file1.acad")
        config.add_recent_file("/test/file2.acad")
        
        recent_files = config.get_recent_files()
        self.assertIn("/test/file1.acad", recent_files)
        self.assertIn("/test/file2.acad", recent_files)


class TestParser(unittest.TestCase):
    """Test script parser functionality"""
    
    def test_lexer_basic(self):
        """Test basic lexing functionality"""
        lexer = ScriptLexer("cube([10, 20, 30]);")
        tokens = lexer.tokenize()
        
        # Should have: keyword, lparen, lbracket, numbers, commas, rbracket, rparen, semicolon, eof
        token_types = [token.type for token in tokens]
        
        self.assertIn(TokenType.KEYWORD, token_types)
        self.assertIn(TokenType.LPAREN, token_types)
        self.assertIn(TokenType.LBRACKET, token_types)
        self.assertIn(TokenType.NUMBER, token_types)
        self.assertIn(TokenType.EOF, token_types)
    
    def test_lexer_comments(self):
        """Test comment parsing"""
        lexer = ScriptLexer("// Single line comment\n/* Multi-line\ncomment */\ncube();")
        tokens = lexer.tokenize()
        
        comment_tokens = [token for token in tokens if token.type == TokenType.COMMENT]
        self.assertEqual(len(comment_tokens), 2)
    
    def test_parser_basic(self):
        """Test basic parsing functionality"""
        script = "cube([10, 20, 30]);"
        ast, errors = parse_script(script)
        
        self.assertEqual(len(errors), 0, f"Parse errors: {errors}")
    
    def test_parser_complex(self):
        """Test parsing complex script"""
        script = '''
        difference() {
            cube([20, 20, 10]);
            translate([0, 0, -1])
                cylinder(h=12, r=3);
        }
        '''
        ast, errors = parse_script(script)
        
        self.assertEqual(len(errors), 0, f"Parse errors: {errors}")


class TestPrimitives(unittest.TestCase):
    """Test primitive shape generation"""
    
    def test_cube_creation(self):
        """Test cube primitive creation"""
        cube_mesh = create_primitive("cube", {"size": [2, 2, 2]})
        
        self.assertEqual(len(cube_mesh.vertices), 8)
        self.assertEqual(len(cube_mesh.faces), 12)
        self.assertIsNotNone(cube_mesh.normals)
    
    def test_sphere_creation(self):
        """Test sphere primitive creation"""
        sphere_mesh = create_primitive("sphere", {"radius": 1.0, "resolution": 16})
        
        self.assertGreater(len(sphere_mesh.vertices), 16)
        self.assertGreater(len(sphere_mesh.faces), 16)
        self.assertIsNotNone(sphere_mesh.normals)
    
    def test_cylinder_creation(self):
        """Test cylinder primitive creation"""
        cylinder_mesh = create_primitive("cylinder", {
            "height": 10,
            "radius": 2,
            "resolution": 12
        })
        
        self.assertGreater(len(cylinder_mesh.vertices), 12)
        self.assertGreater(len(cylinder_mesh.faces), 12)
    
    def test_cone_creation(self):
        """Test cone primitive creation"""
        cone_mesh = create_primitive("cone", {
            "height": 5,
            "radius1": 2,
            "radius2": 0
        })
        
        self.assertGreater(len(cone_mesh.vertices), 6)
        self.assertGreater(len(cone_mesh.faces), 6)
    
    def test_torus_creation(self):
        """Test torus primitive creation"""
        torus_mesh = create_primitive("torus", {
            "major_radius": 3,
            "minor_radius": 1,
            "major_resolution": 12,
            "minor_resolution": 8
        })
        
        expected_vertices = 12 * 8  # major_res * minor_res
        self.assertEqual(len(torus_mesh.vertices), expected_vertices)
    
    def test_available_primitives(self):
        """Test getting available primitives"""
        primitives = primitive_factory.get_available_primitives()
        
        expected_primitives = ['cube', 'box', 'sphere', 'cylinder', 'cone', 'torus']
        for primitive in expected_primitives:
            self.assertIn(primitive, primitives)


class TestCSGOperations(unittest.TestCase):
    """Test CSG operations"""
    
    def setUp(self):
        """Set up test meshes"""
        self.cube1 = create_primitive("cube", {"size": [2, 2, 2]})
        self.cube2 = create_primitive("cube", {"size": [1, 1, 1]})
        self.sphere = create_primitive("sphere", {"radius": 1, "resolution": 16})
    
    def test_union_operation(self):
        """Test union operation"""
        result = union(self.cube1, self.cube2)
        
        self.assertIsNotNone(result.mesh)
        self.assertEqual(len(result.errors), 0)
        self.assertGreater(result.vertex_count, 0)
        self.assertGreater(result.face_count, 0)
    
    def test_difference_operation(self):
        """Test difference operation"""
        result = difference(self.cube1, self.cube2)
        
        self.assertIsNotNone(result.mesh)
        self.assertEqual(len(result.errors), 0)
    
    def test_intersection_operation(self):
        """Test intersection operation"""
        result = intersection(self.cube1, self.sphere)
        
        self.assertIsNotNone(result.mesh)
        self.assertEqual(len(result.errors), 0)
    
    def test_hull_operation(self):
        """Test convex hull operation"""
        result = hull(self.cube1, self.sphere)
        
        self.assertIsNotNone(result.mesh)
        self.assertEqual(len(result.errors), 0)
    
    def test_minkowski_operation(self):
        """Test Minkowski sum operation"""
        result = minkowski(self.cube1, self.sphere)
        
        self.assertIsNotNone(result.mesh)
        self.assertEqual(len(result.errors), 0)


class Test2DShapes(unittest.TestCase):
    """Test 2D shapes and extrusions"""
    
    def test_circle_creation(self):
        """Test 2D circle creation"""
        circle = create_2d_shape("circle", {"radius": 2, "resolution": 16})
        
        self.assertEqual(len(circle.points), 16)
        self.assertTrue(circle.closed)
    
    def test_square_creation(self):
        """Test 2D square creation"""
        square = create_2d_shape("square", {"size": [4, 4]})
        
        self.assertEqual(len(square.points), 4)
        self.assertTrue(square.closed)
    
    def test_linear_extrude(self):
        """Test linear extrusion"""
        mesh = linear_extrude("circle", {"radius": 1, "resolution": 8}, height=5)
        
        self.assertGreater(len(mesh.vertices), 8)
        self.assertGreater(len(mesh.faces), 8)
    
    def test_rotate_extrude(self):
        """Test rotational extrusion"""
        # Create a simple 2D profile for rotation
        mesh = rotate_extrude("square", {"size": [1, 2]}, angle=360, resolution=12)
        
        self.assertGreater(len(mesh.vertices), 12)
        self.assertGreater(len(mesh.faces), 12)


class TestTransformations(unittest.TestCase):
    """Test transformation operations"""
    
    def setUp(self):
        """Set up test mesh"""
        self.cube = create_primitive("cube", {"size": [2, 2, 2], "center": True})
    
    def test_translation(self):
        """Test translation transformation"""
        translated = translate_mesh(self.cube, [5, 0, 0])
        
        # Check that vertices have been translated
        original_center = np.mean(self.cube.vertices, axis=0)
        translated_center = np.mean(translated.vertices, axis=0)
        
        self.assertAlmostEqual(translated_center[0] - original_center[0], 5, places=5)
    
    def test_rotation(self):
        """Test rotation transformation"""
        rotated = rotate_mesh(self.cube, [90, 0, 0])
        
        # After 90-degree rotation around X-axis, Y and Z should be swapped
        self.assertGreater(len(rotated.vertices), 0)
        self.assertEqual(len(rotated.vertices), len(self.cube.vertices))
    
    def test_scaling(self):
        """Test scaling transformation"""
        scaled = scale_mesh(self.cube, [2, 2, 2])
        
        # All vertices should be scaled by factor of 2
        self.assertEqual(len(scaled.vertices), len(self.cube.vertices))
        
        # Check that bounding box is scaled
        original_max = np.max(self.cube.vertices, axis=0)
        scaled_max = np.max(scaled.vertices, axis=0)
        
        np.testing.assert_allclose(scaled_max, original_max * 2, atol=1e-5)
    
    def test_mirroring(self):
        """Test mirroring transformation"""
        mirrored = mirror_mesh(self.cube, [1, 0, 0])  # Mirror across YZ plane
        
        self.assertEqual(len(mirrored.vertices), len(self.cube.vertices))
        
        # X coordinates should be negated
        original_x = self.cube.vertices[:, 0]
        mirrored_x = mirrored.vertices[:, 0]
        
        np.testing.assert_allclose(mirrored_x, -original_x, atol=1e-5)
    
    def test_matrix_creation(self):
        """Test transformation matrix creation"""
        # Test translation matrix
        trans_matrix = create_translation_matrix([1, 2, 3])
        expected = np.eye(4)
        expected[:3, 3] = [1, 2, 3]
        np.testing.assert_array_equal(trans_matrix, expected)
        
        # Test rotation matrix
        rot_matrix = create_rotation_matrix([90, 0, 0])  # 90 degrees around X
        self.assertEqual(rot_matrix.shape, (4, 4))
        
        # Test scale matrix
        scale_matrix = create_scale_matrix([2, 3, 4])
        expected_scale = np.diag([2, 3, 4, 1])
        np.testing.assert_array_equal(scale_matrix, expected_scale)


class TestFileIO(unittest.TestCase):
    """Test file I/O operations"""
    
    def setUp(self):
        """Set up test mesh"""
        self.cube = create_primitive("cube", {"size": [2, 2, 2]})
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
    
    def test_stl_export(self):
        """Test STL export functionality"""
        temp_file = "test_output.stl"
        self.temp_files.append(temp_file)
        
        success = export_mesh(self.cube, temp_file, binary=True)
        self.assertTrue(success)
        self.assertTrue(Path(temp_file).exists())
        self.assertGreater(Path(temp_file).stat().st_size, 0)
    
    def test_obj_export(self):
        """Test OBJ export functionality"""
        temp_file = "test_output.obj"
        self.temp_files.append(temp_file)
        
        success = export_mesh(self.cube, temp_file, precision=6)
        self.assertTrue(success)
        self.assertTrue(Path(temp_file).exists())
        
        # Check that file contains expected OBJ content
        with open(temp_file, 'r') as f:
            content = f.read()
            self.assertIn('v ', content)  # Vertex lines
            self.assertIn('f ', content)  # Face lines
    
    def test_acad_format(self):
        """Test AdvancedCAD native format"""
        temp_file = "test_output.acad"
        self.temp_files.append(temp_file)
        
        # Export
        success = export_mesh(self.cube, temp_file)
        self.assertTrue(success)
        
        # Import back
        import_result = import_mesh(temp_file)
        self.assertEqual(len(import_result.errors), 0)
        self.assertGreater(len(import_result.mesh.vertices), 0)
        
        # Check that imported mesh has same number of vertices
        self.assertEqual(len(import_result.mesh.vertices), len(self.cube.vertices))


class TestPerformance(unittest.TestCase):
    """Test performance characteristics"""
    
    def test_large_mesh_creation(self):
        """Test creation of large meshes"""
        import time
        
        start_time = time.time()
        sphere = create_primitive("sphere", {"radius": 5, "resolution": 64})
        creation_time = time.time() - start_time
        
        self.assertLess(creation_time, 1.0)  # Should create in less than 1 second
        self.assertGreater(len(sphere.vertices), 1000)  # Should have many vertices
    
    def test_csg_performance(self):
        """Test CSG operation performance"""
        import time
        
        cube1 = create_primitive("cube", {"size": [10, 10, 10]})
        cube2 = create_primitive("cube", {"size": [5, 5, 5]})
        
        start_time = time.time()
        result = union(cube1, cube2)
        operation_time = time.time() - start_time
        
        self.assertLess(operation_time, 0.5)  # Should complete in less than 0.5 seconds
        self.assertGreater(result.vertex_count, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple features"""
    
    def test_complete_workflow(self):
        """Test complete modeling workflow"""
        # Create primitives
        base = create_primitive("cube", {"size": [10, 10, 2]})
        hole = create_primitive("cylinder", {"height": 4, "radius": 2, "resolution": 16})
        
        # Apply transformations
        translated_hole = translate_mesh(hole, [0, 0, -1])
        
        # Perform CSG operation
        result = difference(base, translated_hole)
        
        # Verify result
        self.assertEqual(len(result.errors), 0)
        self.assertIsNotNone(result.mesh)
        self.assertGreater(result.vertex_count, 0)
        
        # Export result
        temp_file = "integration_test.stl"
        success = export_mesh(result.mesh, temp_file)
        self.assertTrue(success)
        
        # Clean up
        if Path(temp_file).exists():
            Path(temp_file).unlink()
    
    def test_script_to_mesh(self):
        """Test parsing script and generating mesh"""
        script = '''
        difference() {
            cube([10, 10, 2]);
            translate([0, 0, -1])
                cylinder(h=4, r=2);
        }
        '''
        
        # Parse script
        ast, errors = parse_script(script)
        self.assertEqual(len(errors), 0, f"Parse errors: {errors}")
        
        # Note: Full script execution would require the execution engine
        # This test verifies that parsing works for complex scripts


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
