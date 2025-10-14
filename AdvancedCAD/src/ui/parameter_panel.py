"""
Parameter Panel Widget
Interactive controls for script parameters
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Any


class ParameterControl(QWidget):
    """Base class for parameter controls"""
    
    value_changed = Signal(str, object)  # parameter_name, new_value
    
    def __init__(self, name: str, value: Any, param_type: str):
        super().__init__()
        self.name = name
        self.param_type = param_type
        self.setup_ui(value)
    
    def setup_ui(self, value: Any):
        """Setup the UI for this parameter control"""
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # Label
        label = QLabel(f"{self.name}:")
        label.setMinimumWidth(100)
        layout.addWidget(label)
        
        # Control widget (to be implemented in subclasses)
        self.control_widget = self.create_control_widget(value)
        layout.addWidget(self.control_widget)
        
        # Value display
        self.value_label = QLabel(str(value))
        self.value_label.setMinimumWidth(80)
        layout.addWidget(self.value_label)
    
    def create_control_widget(self, value: Any) -> QWidget:
        """Create the control widget - to be implemented in subclasses"""
        raise NotImplementedError
    
    def get_value(self) -> Any:
        """Get current value - to be implemented in subclasses"""
        raise NotImplementedError
    
    def set_value(self, value: Any):
        """Set current value - to be implemented in subclasses"""
        raise NotImplementedError
    
    def on_value_changed(self, value: Any):
        """Handle value changes"""
        self.value_label.setText(str(value))
        self.value_changed.emit(self.name, value)


class FloatParameterControl(ParameterControl):
    """Control for float parameters"""
    
    def __init__(self, name: str, value: float, min_val: float = -100.0, max_val: float = 100.0):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(name, value, "float")
    
    def create_control_widget(self, value: float) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(self.min_val * 100))
        self.slider.setMaximum(int(self.max_val * 100))
        self.slider.setValue(int(value * 100))
        self.slider.valueChanged.connect(lambda v: self.on_slider_changed(v / 100.0))
        layout.addWidget(self.slider)
        
        # Spin box
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setMinimum(self.min_val)
        self.spinbox.setMaximum(self.max_val)
        self.spinbox.setValue(value)
        self.spinbox.setSingleStep(0.1)
        self.spinbox.setDecimals(2)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)
        layout.addWidget(self.spinbox)
        
        return widget
    
    def on_slider_changed(self, value: float):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self.on_value_changed(value)
    
    def on_spinbox_changed(self, value: float):
        self.slider.blockSignals(True)
        self.slider.setValue(int(value * 100))
        self.slider.blockSignals(False)
        self.on_value_changed(value)
    
    def get_value(self) -> float:
        return self.spinbox.value()
    
    def set_value(self, value: float):
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.slider.setValue(int(value * 100))
        self.spinbox.setValue(value)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)
        self.on_value_changed(value)


class IntParameterControl(ParameterControl):
    """Control for integer parameters"""
    
    def __init__(self, name: str, value: int, min_val: int = -100, max_val: int = 100):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(name, value, "int")
    
    def create_control_widget(self, value: int) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.min_val)
        self.slider.setMaximum(self.max_val)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)
        
        # Spin box
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(self.min_val)
        self.spinbox.setMaximum(self.max_val)
        self.spinbox.setValue(value)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)
        layout.addWidget(self.spinbox)
        
        return widget
    
    def on_slider_changed(self, value: int):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self.on_value_changed(value)
    
    def on_spinbox_changed(self, value: int):
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self.on_value_changed(value)
    
    def get_value(self) -> int:
        return self.spinbox.value()
    
    def set_value(self, value: int):
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)
        self.slider.setValue(value)
        self.spinbox.setValue(value)
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)
        self.on_value_changed(value)


class BoolParameterControl(ParameterControl):
    """Control for boolean parameters"""
    
    def __init__(self, name: str, value: bool):
        super().__init__(name, value, "bool")
    
    def create_control_widget(self, value: bool) -> QWidget:
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(value)
        self.checkbox.toggled.connect(self.on_value_changed)
        return self.checkbox
    
    def get_value(self) -> bool:
        return self.checkbox.isChecked()
    
    def set_value(self, value: bool):
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(value)
        self.checkbox.blockSignals(False)
        self.on_value_changed(value)


class ParameterPanel(QWidget):
    """Panel for displaying and editing script parameters"""
    
    parameter_changed = Signal(str, object)  # parameter_name, new_value
    
    def __init__(self):
        super().__init__()
        self.parameters = {}
        self.controls = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the parameter panel UI"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title = QLabel("Parameters")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)
        
        # Scrollable area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_layout)
        
        scroll_area.setWidget(self.content_widget)
        layout.addWidget(scroll_area)
        
        # Initially show placeholder
        self.show_placeholder()
    
    def show_placeholder(self):
        """Show placeholder when no parameters are available"""
        placeholder = QLabel("No parameters detected\\n\\nParameters will appear here when\\nyou define variables in your script.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #888; padding: 20px;")
        self.content_layout.addWidget(placeholder)
        self.content_layout.addStretch()
    
    def clear_parameters(self):
        """Clear all parameters from the panel"""
        # Remove all widgets from content layout
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.parameters.clear()
        self.controls.clear()
        self.show_placeholder()
    
    def add_parameter(self, name: str, value: Any, param_type: str = None, **kwargs):
        """Add a parameter to the panel"""
        # Remove placeholder if it exists
        if self.content_layout.count() == 2:  # placeholder + stretch
            while self.content_layout.count():
                child = self.content_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        
        # Determine parameter type if not specified
        if param_type is None:
            if isinstance(value, bool):
                param_type = "bool"
            elif isinstance(value, int):
                param_type = "int"
            elif isinstance(value, float):
                param_type = "float"
            else:
                param_type = "string"
        
        # Create appropriate control
        control = None
        if param_type == "float":
            min_val = kwargs.get("min", -100.0)
            max_val = kwargs.get("max", 100.0)
            control = FloatParameterControl(name, value, min_val, max_val)
        elif param_type == "int":
            min_val = kwargs.get("min", -100)
            max_val = kwargs.get("max", 100)
            control = IntParameterControl(name, value, min_val, max_val)
        elif param_type == "bool":
            control = BoolParameterControl(name, value)
        
        if control:
            # Connect signal
            control.value_changed.connect(self.on_parameter_changed)
            
            # Add to layout
            self.content_layout.addWidget(control)
            
            # Store references
            self.parameters[name] = value
            self.controls[name] = control
        
        # Add stretch at the end
        self.content_layout.addStretch()
    
    def on_parameter_changed(self, name: str, value: Any):
        """Handle parameter value changes"""
        self.parameters[name] = value
        self.parameter_changed.emit(name, value)
    
    def get_parameter(self, name: str) -> Any:
        """Get parameter value"""
        return self.parameters.get(name)
    
    def set_parameter(self, name: str, value: Any):
        """Set parameter value"""
        if name in self.controls:
            self.controls[name].set_value(value)
    
    def update_parameters_from_script(self, script_text: str):
        """Update parameters based on script analysis"""
        # This is a simplified implementation
        # In a full implementation, this would parse the script to find variable definitions
        
        # For demo purposes, add some sample parameters
        import re
        
        self.clear_parameters()
        
        # Find variable assignments in the script
        pattern = r'^\\s*([a-zA-Z_][a-zA-Z0-9_]*)\\s*=\\s*([^;]+);'
        matches = re.findall(pattern, script_text, re.MULTILINE)
        
        for var_name, var_value in matches:
            var_value = var_value.strip()
            
            # Try to parse the value
            try:
                if var_value in ['true', 'false']:
                    value = var_value == 'true'
                    self.add_parameter(var_name, value)
                elif '.' in var_value:
                    value = float(var_value)
                    self.add_parameter(var_name, value, min=-50.0, max=50.0)
                else:
                    value = int(var_value)
                    self.add_parameter(var_name, value, min=-50, max=50)
            except ValueError:
                # Skip complex expressions for now
                pass
