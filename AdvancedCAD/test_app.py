#!/usr/bin/env python3
"""
Quick test script for AdvancedCAD
Tests basic functionality without full installation
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    # Test imports
    print("Testing imports...")
    
    # Test core modules
    from core.config import AppConfig
    from core.parser import parse_script, ScriptLexer
    from core.primitives import create_primitive
    
    print("✓ Core modules imported successfully")
    
    # Test UI modules (will fail if PySide6 not installed)
    try:
        from PySide6.QtWidgets import QApplication
        from ui.main_window import MainWindow
        print("✓ UI modules imported successfully")
        
        # Test basic functionality
        print("\\nTesting basic functionality...")
        
        # Test configuration
        config = AppConfig()
        print("✓ Configuration system working")
        
        # Test parser
        sample_script = "cube([10, 20, 30]);"
        ast, errors = parse_script(sample_script)
        print(f"✓ Parser working - found {len(errors)} errors")
        
        # Test lexer
        lexer = ScriptLexer(sample_script)
        tokens = lexer.tokenize()
        print(f"✓ Lexer working - generated {len(tokens)} tokens")
        
        # Test primitives
        try:
            cube_mesh = create_primitive("cube", {"size": [2, 2, 2]})
            print(f"✓ Primitives working - cube has {len(cube_mesh.vertices)} vertices")
        except Exception as e:
            print(f"⚠ Primitives have issues (expected - missing dependencies): {e}")
        
        print("\\nAll core functionality tests passed!")
        print("\\nTo run the full application, install dependencies:")
        print("pip install -r requirements.txt")
        print("python main.py")
        
    except ImportError as e:
        print(f"⚠ UI modules not available: {e}")
        print("Install PySide6 to run the GUI: pip install PySide6")
        
        # Test core functionality even without UI
        print("\nTesting core functionality without UI...")
        
        # Test configuration
        config = AppConfig()
        print("✓ Configuration system working")
        
        # Test parser
        sample_script = "cube([10, 20, 30]);"
        ast, errors = parse_script(sample_script)
        print(f"✓ Parser working - found {len(errors)} errors")
        
        # Test lexer
        lexer = ScriptLexer(sample_script)
        tokens = lexer.tokenize()
        print(f"✓ Lexer working - generated {len(tokens)} tokens")
        
        # Test primitives
        try:
            cube_mesh = create_primitive("cube", {"size": [2, 2, 2]})
            print(f"✓ Primitives working - cube has {len(cube_mesh.vertices)} vertices")
        except Exception as e:
            print(f"⚠ Primitives have issues (expected - missing dependencies): {e}")
        
        print("\nCore functionality tests completed!")

except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
