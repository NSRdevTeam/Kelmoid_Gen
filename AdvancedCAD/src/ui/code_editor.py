"""
Code Editor Widget for AdvancedCAD
Advanced text editor with syntax highlighting, auto-completion, and IDE features
"""

from PySide6.QtWidgets import QTextEdit, QPlainTextEdit, QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QFont, QColor, QPalette, QPainter, QTextCharFormat, QSyntaxHighlighter, QTextDocument, QTextFormat


class LineNumberArea(QWidget):
    """Widget for displaying line numbers"""
    
    def __init__(self, editor):
        super().__init__(editor)
        self.code_editor = editor
    
    def sizeHint(self):
        return self.code_editor.line_number_area_width()
    
    def paintEvent(self, event):
        self.code_editor.line_number_area_paint_event(event)


class AdvancedCADSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for AdvancedCAD/OpenSCAD language"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_highlighting_rules()
    
    def setup_highlighting_rules(self):
        """Setup syntax highlighting rules"""
        # Keywords
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor(86, 156, 214))  # Blue
        keyword_format.setFontWeight(QFont.Bold)
        
        keywords = [
            'cube', 'sphere', 'cylinder', 'cone', 'torus', 'polyhedron',
            'circle', 'square', 'polygon', 'text',
            'union', 'difference', 'intersection', 'hull', 'minkowski',
            'translate', 'rotate', 'scale', 'mirror', 'color',
            'linear_extrude', 'rotate_extrude',
            'module', 'function', 'if', 'else', 'for', 'let',
            'true', 'false', 'undef'
        ]
        
        self.keyword_rules = []
        for keyword in keywords:
            pattern = f"\\b{keyword}\\b"
            self.keyword_rules.append((pattern, keyword_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(181, 206, 168))  # Light green
        self.number_pattern = r'\b\d+\.?\d*\b'
        
        # Strings
        string_format = QTextCharFormat()
        string_format.setForeground(QColor(206, 145, 120))  # Orange
        self.string_patterns = [r'".*"', r"'.*'"]
        
        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor(106, 153, 85))  # Green
        comment_format.setFontItalic(True)
        self.comment_patterns = [r'//.*', r'/\*.*\*/']
        
        # Store formats
        self.formats = {
            'keyword': keyword_format,
            'number': number_format,
            'string': string_format,
            'comment': comment_format
        }
    
    def highlightBlock(self, text):
        """Highlight a block of text"""
        import re
        
        # Highlight keywords
        for pattern, format in self.keyword_rules:
            for match in re.finditer(pattern, text):
                start, length = match.span()
                self.setFormat(start, length, format)
        
        # Highlight numbers
        for match in re.finditer(self.number_pattern, text):
            start, length = match.span()
            self.setFormat(start, length, self.formats['number'])
        
        # Highlight strings
        for pattern in self.string_patterns:
            for match in re.finditer(pattern, text):
                start, length = match.span()
                self.setFormat(start, length, self.formats['string'])
        
        # Highlight comments (should be last to override other formatting)
        for pattern in self.comment_patterns:
            for match in re.finditer(pattern, text):
                start, length = match.span()
                self.setFormat(start, length, self.formats['comment'])


class CodeEditor(QPlainTextEdit):
    """Enhanced code editor widget"""
    
    def __init__(self):
        super().__init__()
        
        # Create line number area
        self.line_number_area = LineNumberArea(self)
        
        # Setup syntax highlighting
        self.highlighter = AdvancedCADSyntaxHighlighter(self.document())
        
        # Connect signals
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.highlight_current_line)
        
        # Initial setup
        self.update_line_number_area_width(0)
        self.highlight_current_line()
        
        # Set editor properties
        self.setup_editor()
    
    def setup_editor(self):
        """Setup editor properties"""
        # Font
        font = QFont("Consolas", 10)
        font.setFixedPitch(True)
        self.setFont(font)
        
        # Colors and styling
        palette = self.palette()
        palette.setColor(QPalette.Base, QColor(30, 30, 30))  # Dark background
        palette.setColor(QPalette.Text, QColor(220, 220, 220))  # Light text
        self.setPalette(palette)
        
        # Tab settings
        self.setTabStopDistance(40)  # 4 spaces
        
        # Line wrapping
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        
        # Auto-indentation and other editor behaviors would go here
    
    def line_number_area_width(self):
        """Calculate width needed for line numbers"""
        digits = 1
        max_num = max(1, self.blockCount())
        while max_num >= 10:
            max_num //= 10
            digits += 1
        
        space = 3 + self.fontMetrics().horizontalAdvance('9') * digits
        return space
    
    def update_line_number_area_width(self, new_block_count):
        """Update line number area width"""
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)
    
    def update_line_number_area(self, rect, dy):
        """Update line number area"""
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), 
                                       self.line_number_area.width(), 
                                       rect.height())
        
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)
    
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), 
                  self.line_number_area_width(), cr.height())
        )
    
    def line_number_area_paint_event(self, event):
        """Paint line numbers"""
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor(45, 45, 45))  # Slightly lighter background
        
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()
        
        # Set text color for line numbers
        painter.setPen(QColor(128, 128, 128))
        
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.drawText(0, int(top), 
                               self.line_number_area.width() - 3, 
                               self.fontMetrics().height(),
                               Qt.AlignRight, number)
            
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1
    
    def highlight_current_line(self):
        """Highlight the current line"""
        extra_selections = []
        
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            
            line_color = QColor(68, 68, 68)  # Dark gray highlight
            selection.format.setBackground(line_color)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extra_selections.append(selection)
        
        self.setExtraSelections(extra_selections)
    
    def keyPressEvent(self, event):
        """Handle key press events for auto-indentation and other features"""
        # Auto-indentation on Enter
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            cursor = self.textCursor()
            
            # Get current line
            cursor.select(cursor.LineUnderCursor)
            line = cursor.selectedText()
            
            # Calculate indentation of current line
            indent = 0
            for char in line:
                if char == ' ':
                    indent += 1
                elif char == '\t':
                    indent += 4  # Assume tab = 4 spaces
                else:
                    break
            
            # Check if we need to increase indentation
            stripped_line = line.strip()
            if stripped_line.endswith('{') or stripped_line.endswith('('):
                indent += 4
            
            # Insert new line with proper indentation
            cursor.movePosition(cursor.EndOfLine)
            self.setTextCursor(cursor)
            super().keyPressEvent(event)
            
            # Add indentation
            self.insertPlainText(' ' * indent)
            return
        
        # Auto-close brackets and parentheses
        auto_pairs = {
            '(': ')',
            '[': ']',
            '{': '}',
            '"': '"',
            "'": "'"
        }
        
        if event.text() in auto_pairs:
            cursor = self.textCursor()
            self.insertPlainText(event.text() + auto_pairs[event.text()])
            cursor.movePosition(cursor.Left)
            self.setTextCursor(cursor)
            return
        
        super().keyPressEvent(event)
