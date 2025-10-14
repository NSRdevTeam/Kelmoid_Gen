"""
Application configuration management
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml

try:
    from PySide6.QtCore import QSettings
    HAS_PYSIDE6 = True
except ImportError:
    QSettings = None
    HAS_PYSIDE6 = False


class AppConfig:
    """Manages application configuration and settings"""
    
    def __init__(self):
        self.settings = QSettings() if HAS_PYSIDE6 else None
        self.config_file = Path.home() / ".advancedcad" / "config.yaml"
        self.config_file.parent.mkdir(exist_ok=True)
        
        # Default configuration
        self.defaults = {
            # UI Settings
            "ui": {
                "theme": "dark",
                "font_size": 10,
                "editor_font": "Consolas",
                "show_line_numbers": True,
                "auto_save": True,
                "auto_save_interval": 30,  # seconds
            },
            
            # Rendering Settings
            "rendering": {
                "anti_aliasing": True,
                "show_axes": True,
                "background_color": [0.2, 0.2, 0.3],
                "grid_enabled": True,
                "wireframe_mode": False,
                "lighting_enabled": True,
            },
            
            # Performance Settings
            "performance": {
                "max_triangles": 1000000,
                "use_multi_threading": True,
                "cache_enabled": True,
                "progressive_rendering": True,
            },
            
            # File Settings
            "files": {
                "recent_files": [],
                "max_recent_files": 10,
                "default_export_format": "stl",
                "auto_backup": True,
            },
            
            # Script Settings
            "script": {
                "auto_complete": True,
                "bracket_matching": True,
                "syntax_highlighting": True,
                "auto_render_on_change": False,
                "render_delay": 500,  # milliseconds
            }
        }
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                # Merge with defaults
                return self._merge_config(self.defaults, config)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.defaults.copy()
        else:
            return self.defaults.copy()
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'ui.theme')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        self.save_config()
    
    def _merge_config(self, defaults: Dict, user_config: Dict) -> Dict:
        """Recursively merge user config with defaults"""
        result = defaults.copy()
        
        for key, value in user_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def add_recent_file(self, file_path: str):
        """Add file to recent files list"""
        recent_files = self.get("files.recent_files", [])
        
        # Remove if already exists
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # Add to beginning
        recent_files.insert(0, file_path)
        
        # Limit to max recent files
        max_recent = self.get("files.max_recent_files", 10)
        recent_files = recent_files[:max_recent]
        
        self.set("files.recent_files", recent_files)
    
    def get_recent_files(self) -> list:
        """Get list of recent files"""
        return self.get("files.recent_files", [])
