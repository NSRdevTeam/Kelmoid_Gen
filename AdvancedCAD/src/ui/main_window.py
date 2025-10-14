"""
Main Window for AdvancedCAD Application
Provides the primary user interface with docking panels and modern design
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QDockWidget, QMenuBar, QToolBar, QStatusBar,
    QLabel, QProgressBar, QApplication, QMessageBox, QFileDialog
)
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QFont, QPixmap
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize


# Import our custom widgets
from .code_editor import CodeEditor
from .parameter_panel import ParameterPanel
from .viewport_3d import Viewport3D
from .project_explorer import ProjectExplorer
from .console_widget import ConsoleWidget

# Import core functionality
from core.config import AppConfig
from core.parser import parse_script


class RenderThread(QThread):
    """Background thread for 3D rendering operations"""
    
    render_complete = Signal(object)  # Emits mesh data when rendering is complete
    render_error = Signal(str)        # Emits error message if rendering fails
    
    def __init__(self, script_text: str):
        super().__init__()
        self.script_text = script_text
    
    def run(self):
        """Run the rendering process in background"""
        try:
            # Parse the script
            ast, errors = parse_script(self.script_text)
            
            if errors:
                error_msg = "\n".join([f"Line {e.line}: {e.message}" for e in errors])
                self.render_error.emit(f"Parse errors:\n{error_msg}")
                return
            
            # For now, emit a simple result
            # In full implementation, this would generate the 3D mesh
            self.render_complete.emit({"ast": ast, "vertex_count": 0, "face_count": 0})
            
        except Exception as e:
            self.render_error.emit(f"Rendering error: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize configuration
        self.config = AppConfig()
        
        # Initialize variables
        self.current_file = None
        self.is_modified = False
        self.render_thread = None
        
        # Setup UI
        self.setup_ui()
        self.setup_menus()
        self.setup_toolbars()
        self.setup_status_bar()
        self.setup_connections()
        
        # Apply configuration
        self.apply_config()
        
        # Set window properties
        self.setWindowTitle("AdvancedCAD - Next-Generation 3D Modeling")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)
        
        # Start with a sample script
        self.load_sample_script()
    
    def setup_ui(self):
        """Setup the main user interface"""
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)
        
        # Create left panel (code editor + project explorer)
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setMinimumWidth(400)
        
        # Code editor
        self.code_editor = CodeEditor()
        left_splitter.addWidget(self.code_editor)
        
        # Project explorer (docked widget)
        self.setup_project_explorer()
        
        # Create right panel (3D viewport + parameter panel)
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setMinimumWidth(600)
        
        # 3D Viewport
        self.viewport_3d = Viewport3D()
        
        # If the custom viewport is not a QWidget (e.g., GL stack unavailable), fallback to a placeholder widget
        if not isinstance(self.viewport_3d, QWidget):
            placeholder = QWidget()
            placeholder_layout = QVBoxLayout()
            placeholder.setLayout(placeholder_layout)
            info_label = QLabel("3D Viewport Unavailable")
            info_label.setAlignment(Qt.AlignCenter)
            placeholder_layout.addWidget(info_label)
            self.viewport_3d = placeholder
        
        # Wrap viewport in a QWidget container to ensure compatibility with QSplitter
        viewport_container = QWidget()
        viewport_container.setLayout(QVBoxLayout())
        viewport_container.layout().setContentsMargins(0, 0, 0, 0)
        viewport_container.layout().addWidget(self.viewport_3d)
        right_splitter.addWidget(viewport_container)
        
        # Parameter panel (docked widget)
        self.setup_parameter_panel()
        
        # Add panels to main splitter
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        
        # Set splitter proportions
        main_splitter.setSizes([500, 900])  # 35% left, 65% right
        
        # Setup console as docked widget
        self.setup_console()
    
    def setup_project_explorer(self):
        """Setup project explorer dock widget"""
        self.project_dock = QDockWidget("Project Explorer", self)
        self.project_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.project_explorer = ProjectExplorer()
        self.project_dock.setWidget(self.project_explorer)
        
        self.addDockWidget(Qt.LeftDockWidgetArea, self.project_dock)
    
    def setup_parameter_panel(self):
        """Setup parameter panel dock widget"""
        self.parameter_dock = QDockWidget("Parameters", self)
        self.parameter_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        
        self.parameter_panel = ParameterPanel()
        self.parameter_dock.setWidget(self.parameter_panel)
        
        self.addDockWidget(Qt.RightDockWidgetArea, self.parameter_dock)
    
    def setup_console(self):
        """Setup console dock widget"""
        self.console_dock = QDockWidget("Console", self)
        self.console_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        
        self.console = ConsoleWidget()
        self.console_dock.setWidget(self.console)
        
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console_dock)
        
        # Initially hide the console
        self.console_dock.hide()
    
    def setup_menus(self):
        """Setup application menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # New file
        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.setStatusTip("Create a new file")
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)
        
        # Open file
        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setStatusTip("Open an existing file")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        # Recent files submenu
        self.recent_files_menu = file_menu.addMenu("Recent Files")
        self.update_recent_files_menu()
        
        file_menu.addSeparator()
        
        # Save file
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.setStatusTip("Save the current file")
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        # Save As
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.setStatusTip("Save the current file with a new name")
        save_as_action.triggered.connect(self.save_file_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # Export submenu
        export_menu = file_menu.addMenu("Export")
        
        export_stl_action = QAction("Export STL...", self)
        export_stl_action.setStatusTip("Export model as STL file")
        export_stl_action.triggered.connect(lambda: self.export_model("stl"))
        export_menu.addAction(export_stl_action)
        
        export_obj_action = QAction("Export OBJ...", self)
        export_obj_action.setStatusTip("Export model as OBJ file")
        export_obj_action.triggered.connect(lambda: self.export_model("obj"))
        export_menu.addAction(export_obj_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        # Undo/Redo
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.code_editor.undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.code_editor.redo)
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        # Cut/Copy/Paste
        cut_action = QAction("Cu&t", self)
        cut_action.setShortcut(QKeySequence.Cut)
        cut_action.triggered.connect(self.code_editor.cut)
        edit_menu.addAction(cut_action)
        
        copy_action = QAction("&Copy", self)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.triggered.connect(self.code_editor.copy)
        edit_menu.addAction(copy_action)
        
        paste_action = QAction("&Paste", self)
        paste_action.setShortcut(QKeySequence.Paste)
        paste_action.triggered.connect(self.code_editor.paste)
        edit_menu.addAction(paste_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        # Toggle docks
        view_menu.addAction(self.project_dock.toggleViewAction())
        view_menu.addAction(self.parameter_dock.toggleViewAction())
        view_menu.addAction(self.console_dock.toggleViewAction())
        
        view_menu.addSeparator()
        
        # Viewport options
        wireframe_action = QAction("Wireframe Mode", self)
        wireframe_action.setCheckable(True)
        wireframe_action.toggled.connect(self.toggle_wireframe)
        view_menu.addAction(wireframe_action)
        
        # Design menu
        design_menu = menubar.addMenu("&Design")
        
        # Render
        render_action = QAction("&Render", self)
        render_action.setShortcut("F5")
        render_action.setStatusTip("Render the 3D model")
        render_action.triggered.connect(self.render_model)
        design_menu.addAction(render_action)
        
        # Quick render
        quick_render_action = QAction("&Quick Render", self)
        quick_render_action.setShortcut("F6")
        quick_render_action.setStatusTip("Quick preview render")
        quick_render_action.triggered.connect(self.quick_render)
        design_menu.addAction(quick_render_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        # Documentation
        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut("F1")
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
        
        help_menu.addSeparator()
        
        # About
        about_action = QAction("&About AdvancedCAD", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbars(self):
        """Setup application toolbars"""
        
        # Main toolbar
        main_toolbar = self.addToolBar("Main")
        main_toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        # File operations
        new_action = QAction("New", self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_file)
        main_toolbar.addAction(new_action)
        
        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_file)
        main_toolbar.addAction(open_action)
        
        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_file)
        main_toolbar.addAction(save_action)
        
        main_toolbar.addSeparator()
        
        # Render operations
        render_action = QAction("Render", self)
        render_action.setShortcut("F5")
        render_action.triggered.connect(self.render_model)
        main_toolbar.addAction(render_action)
        
        quick_render_action = QAction("Quick", self)
        quick_render_action.setShortcut("F6")
        quick_render_action.triggered.connect(self.quick_render)
        main_toolbar.addAction(quick_render_action)
        
        # View toolbar
        view_toolbar = self.addToolBar("View")
        
        # View mode group
        view_group = QActionGroup(self)
        
        solid_action = QAction("Solid", self)
        solid_action.setCheckable(True)
        solid_action.setChecked(True)
        solid_action.triggered.connect(lambda: self.set_view_mode("solid"))
        view_group.addAction(solid_action)
        view_toolbar.addAction(solid_action)
        
        wireframe_action = QAction("Wireframe", self)
        wireframe_action.setCheckable(True)
        wireframe_action.triggered.connect(lambda: self.set_view_mode("wireframe"))
        view_group.addAction(wireframe_action)
        view_toolbar.addAction(wireframe_action)
        
        transparent_action = QAction("Transparent", self)
        transparent_action.setCheckable(True)
        transparent_action.triggered.connect(lambda: self.set_view_mode("transparent"))
        view_group.addAction(transparent_action)
        view_toolbar.addAction(transparent_action)
    
    def setup_status_bar(self):
        """Setup status bar"""
        status_bar = self.statusBar()
        
        # Status label
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_bar.addPermanentWidget(self.progress_bar)
        
        # Model info
        self.model_info_label = QLabel("No model")
        status_bar.addPermanentWidget(self.model_info_label)
    
    def setup_connections(self):
        """Setup signal/slot connections"""
        
        # Code editor changes
        self.code_editor.textChanged.connect(self.on_text_changed)
        
        # Parameter panel changes
        self.parameter_panel.parameter_changed.connect(self.on_parameter_changed)
        
        # Auto-render timer
        self.auto_render_timer = QTimer()
        self.auto_render_timer.setSingleShot(True)
        self.auto_render_timer.timeout.connect(self.auto_render)
        
        # Console log connections
        self.console.message_logged.connect(self.on_console_message)
    
    def apply_config(self):
        """Apply configuration settings"""
        
        # Editor settings
        font_family = self.config.get("ui.editor_font", "Consolas")
        font_size = self.config.get("ui.font_size", 10)
        font = QFont(font_family, font_size)
        self.code_editor.setFont(font)
        
        # Auto-render settings
        auto_render = self.config.get("script.auto_render_on_change", False)
        if auto_render:
            render_delay = self.config.get("script.render_delay", 500)
            self.auto_render_timer.setInterval(render_delay)
    
    def load_sample_script(self):
        """Load a sample script to demonstrate functionality"""
        sample_script = '''// AdvancedCAD Sample Script
// Create a simple mechanical part

// Define parameters
part_width = 20;
part_height = 10;
part_depth = 30;
hole_radius = 3;

// Main body
difference() {
    // Outer shape
    cube([part_width, part_height, part_depth], center=true);
    
    // Mounting holes
    translate([6, 0, 0])
        cylinder(h=part_height+2, r=hole_radius, center=true);
    
    translate([-6, 0, 0])
        cylinder(h=part_height+2, r=hole_radius, center=true);
}

// Add reinforcement ribs
for (i = [-1, 1]) {
    translate([i * 8, part_height/2, 0])
        cube([2, 2, part_depth], center=true);
}'''
        
        self.code_editor.setPlainText(sample_script)
        self.is_modified = False
        self.update_window_title()
    
    def update_recent_files_menu(self):
        """Update recent files menu"""
        self.recent_files_menu.clear()
        recent_files = self.config.get_recent_files()
        
        for file_path in recent_files:
            if Path(file_path).exists():
                action = QAction(str(file_path), self)
                action.triggered.connect(lambda checked, path=file_path: self.open_recent_file(path))
                self.recent_files_menu.addAction(action)
        
        if not recent_files:
            no_recent_action = QAction("No recent files", self)
            no_recent_action.setEnabled(False)
            self.recent_files_menu.addAction(no_recent_action)
    
    def update_window_title(self):
        """Update window title based on current file and modification state"""
        title = "AdvancedCAD"
        
        if self.current_file:
            title += f" - {self.current_file}"
        else:
            title += " - Untitled"
        
        if self.is_modified:
            title += " *"
        
        self.setWindowTitle(title)
    
    # File operations
    def new_file(self):
        """Create a new file"""
        if self.check_save_changes():
            self.code_editor.clear()
            self.current_file = None
            self.is_modified = False
            self.update_window_title()
            self.status_label.setText("New file created")
    
    def open_file(self):
        """Open an existing file"""
        if not self.check_save_changes():
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "AdvancedCAD Files (*.acad);;OpenSCAD Files (*.scad);;All Files (*)"
        )
        
        if file_path:
            self.load_file(file_path)
    
    def open_recent_file(self, file_path: str):
        """Open a recent file"""
        if not self.check_save_changes():
            return
        
        self.load_file(file_path)
    
    def load_file(self, file_path: str):
        """Load a file from disk"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            self.code_editor.setPlainText(content)
            self.current_file = file_path
            self.is_modified = False
            self.update_window_title()
            
            # Add to recent files
            self.config.add_recent_file(file_path)
            self.update_recent_files_menu()
            
            self.status_label.setText(f"Loaded: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open file: {str(e)}")
    
    def save_file(self) -> bool:
        """Save the current file"""
        if self.current_file:
            return self.save_to_file(self.current_file)
        else:
            return self.save_file_as()
    
    def save_file_as(self) -> bool:
        """Save the current file with a new name"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", "",
            "AdvancedCAD Files (*.acad);;OpenSCAD Files (*.scad);;All Files (*)"
        )
        
        if file_path:
            return self.save_to_file(file_path)
        
        return False
    
    def save_to_file(self, file_path: str) -> bool:
        """Save content to specific file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(self.code_editor.toPlainText())
            
            self.current_file = file_path
            self.is_modified = False
            self.update_window_title()
            
            # Add to recent files
            self.config.add_recent_file(file_path)
            self.update_recent_files_menu()
            
            self.status_label.setText(f"Saved: {file_path}")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save file: {str(e)}")
            return False
    
    def check_save_changes(self) -> bool:
        """Check if user wants to save changes before proceeding"""
        if not self.is_modified:
            return True
        
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "The document has unsaved changes. Do you want to save them?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )
        
        if reply == QMessageBox.Save:
            return self.save_file()
        elif reply == QMessageBox.Discard:
            return True
        else:
            return False
    
    # Rendering operations
    def render_model(self):
        """Render the 3D model"""
        self.start_render(full_quality=True)
    
    def quick_render(self):
        """Quick preview render"""
        self.start_render(full_quality=False)
    
    def start_render(self, full_quality: bool = True):
        """Start rendering in background thread"""
        if self.render_thread and self.render_thread.isRunning():
            self.render_thread.quit()
            self.render_thread.wait()
        
        script_text = self.code_editor.toPlainText()
        
        self.render_thread = RenderThread(script_text)
        self.render_thread.render_complete.connect(self.on_render_complete)
        self.render_thread.render_error.connect(self.on_render_error)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Rendering...")
        
        self.render_thread.start()
    
    def auto_render(self):
        """Auto-render triggered by timer"""
        if self.config.get("script.auto_render_on_change", False):
            self.quick_render()
    
    def on_render_complete(self, result: Dict[str, Any]):
        """Handle successful render completion"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Render complete")
        
        # For now, create a simple demo mesh to show the viewport works
        try:
            from core.primitives import create_primitive
            
            # Create a demo mesh based on script content
            script_content = self.code_editor.toPlainText().lower()
            
            if "sphere" in script_content:
                demo_mesh = create_primitive('sphere', {'radius': 8, 'resolution': 32})
            elif "cylinder" in script_content:
                demo_mesh = create_primitive('cylinder', {'height': 15, 'radius': 5, 'resolution': 24})
            elif "torus" in script_content:
                demo_mesh = create_primitive('torus', {'major_radius': 8, 'minor_radius': 2})
            else:
                # Default to cube
                demo_mesh = create_primitive('cube', {'size': [12, 8, 10]})
            
            # Update the viewport if available
            if hasattr(self.viewport_3d, 'set_mesh'):
                self.viewport_3d.set_mesh(demo_mesh)
            
            # Update model info
            vertex_count = len(demo_mesh.vertices) if getattr(demo_mesh, 'vertices', None) is not None else 0
            face_count = len(demo_mesh.faces) if getattr(demo_mesh, 'faces', None) is not None else 0
            self.model_info_label.setText(f"Vertices: {vertex_count}, Faces: {face_count}")
            
        except Exception as e:
            print(f"Error creating demo mesh: {e}")
            # Update model info with result data
            vertex_count = result.get("vertex_count", 0)
            face_count = result.get("face_count", 0)
            self.model_info_label.setText(f"Vertices: {vertex_count}, Faces: {face_count}")
        
        self.console.log_message("Render completed successfully", "info")
    
    def on_render_error(self, error_message: str):
        """Handle render error"""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Render error")
        
        self.console.log_message(f"Render error: {error_message}", "error")
        self.console_dock.show()  # Show console on error
    
    # Event handlers
    def on_text_changed(self):
        """Handle text changes in code editor"""
        self.is_modified = True
        self.update_window_title()
        
        # Start auto-render timer if enabled
        if self.config.get("script.auto_render_on_change", False):
            self.auto_render_timer.start()
    
    def on_parameter_changed(self, param_name: str, param_value: Any):
        """Handle parameter changes"""
        # Update code editor with new parameter value
        # This is a simplified implementation
        self.console.log_message(f"Parameter changed: {param_name} = {param_value}", "info")
    
    def on_console_message(self, message: str, level: str):
        """Handle console log messages"""
        if level == "error":
            self.status_label.setText("Error - see console")
    
    # View operations
    def toggle_wireframe(self, enabled: bool):
        """Toggle wireframe view mode"""
        if hasattr(self.viewport_3d, 'set_wireframe_mode'):
            self.viewport_3d.set_wireframe_mode(enabled)
    
    def set_view_mode(self, mode: str):
        """Set viewport view mode"""
        if hasattr(self.viewport_3d, 'set_view_mode'):
            self.viewport_3d.set_view_mode(mode)
    
    def export_model(self, format: str):
        """Export model in specified format"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Export {format.upper()}", "",
            f"{format.upper()} Files (*.{format});;All Files (*)"
        )
        
        if file_path:
            # In full implementation, this would export the current model
            self.console.log_message(f"Export to {format.upper()} not yet implemented", "warning")
    
    # Help operations
    def show_documentation(self):
        """Show documentation"""
        QMessageBox.information(
            self, "Documentation",
            "Documentation is not yet available. Please check the GitHub repository."
        )
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About AdvancedCAD",
            "AdvancedCAD v1.0\n\n"
            "Next-generation 3D modeling software inspired by OpenSCAD "
            "but with significant improvements in user experience, performance, and functionality.\n\n"
            "Built with Python and PySide6."
        )
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.check_save_changes():
            # Stop render thread if running
            if self.render_thread and self.render_thread.isRunning():
                self.render_thread.quit()
                self.render_thread.wait()
            
            event.accept()
        else:
            event.ignore()
