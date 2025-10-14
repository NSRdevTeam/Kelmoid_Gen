"""
Script parser for AdvancedCAD modeling language
Improved version of OpenSCAD syntax with better error handling
"""

import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math


class TokenType(Enum):
    """Token types for the parser"""
    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"
    KEYWORD = "KEYWORD"
    OPERATOR = "OPERATOR"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    SEMICOLON = "SEMICOLON"
    COMMA = "COMMA"
    EQUALS = "EQUALS"
    NEWLINE = "NEWLINE"
    EOF = "EOF"
    COMMENT = "COMMENT"


@dataclass
class Token:
    """Token data structure"""
    type: TokenType
    value: str
    line: int
    column: int


@dataclass
class ParseError:
    """Parse error information"""
    message: str
    line: int
    column: int
    severity: str = "error"  # "error", "warning", "info"


class ASTNode:
    """Base class for AST nodes"""
    def __init__(self, line: int = 0, column: int = 0):
        self.line = line
        self.column = column


class PrimitiveNode(ASTNode):
    """Node for primitive shapes"""
    def __init__(self, name: str, params: Dict[str, Any], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.params = params


class TransformNode(ASTNode):
    """Node for transformations"""
    def __init__(self, name: str, params: Dict[str, Any], children: List[ASTNode], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.params = params
        self.children = children


class OperationNode(ASTNode):
    """Node for CSG operations"""
    def __init__(self, operation: str, children: List[ASTNode], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.operation = operation
        self.children = children


class ModuleNode(ASTNode):
    """Node for module definitions"""
    def __init__(self, name: str, params: List[str], body: List[ASTNode], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.params = params
        self.body = body


class AssignmentNode(ASTNode):
    """Node for variable assignments"""
    def __init__(self, name: str, value: Any, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.value = value


class ScriptLexer:
    """Lexical analyzer for the modeling script"""
    
    KEYWORDS = {
        'cube', 'sphere', 'cylinder', 'cone', 'torus', 'polyhedron',
        'circle', 'square', 'polygon', 'text',
        'union', 'difference', 'intersection', 'hull', 'minkowski',
        'translate', 'rotate', 'scale', 'mirror', 'color',
        'linear_extrude', 'rotate_extrude',
        'module', 'function', 'if', 'else', 'for', 'let',
        'true', 'false', 'undef'
    }
    
    OPERATORS = {
        '+', '-', '*', '/', '%', '^',
        '==', '!=', '<', '>', '<=', '>=',
        '&&', '||', '!',
        '?', ':'
    }
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        self.errors = []
    
    def error(self, message: str):
        """Add error to error list"""
        self.errors.append(ParseError(message, self.line, self.column))
    
    def current_char(self) -> Optional[str]:
        """Get current character"""
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        """Peek at character at offset"""
        peek_pos = self.pos + offset
        if peek_pos >= len(self.text):
            return None
        return self.text[peek_pos]
    
    def advance(self):
        """Advance position and update line/column"""
        if self.pos < len(self.text) and self.text[self.pos] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.pos += 1
    
    def skip_whitespace(self):
        """Skip whitespace characters except newlines"""
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def read_number(self) -> Token:
        """Read numeric token"""
        start_line, start_column = self.line, self.column
        result = ''
        
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            result += self.current_char()
            self.advance()
        
        # Handle scientific notation
        if self.current_char() and self.current_char().lower() == 'e':
            result += self.current_char()
            self.advance()
            if self.current_char() and self.current_char() in '+-':
                result += self.current_char()
                self.advance()
            while self.current_char() and self.current_char().isdigit():
                result += self.current_char()
                self.advance()
        
        return Token(TokenType.NUMBER, result, start_line, start_column)
    
    def read_string(self) -> Token:
        """Read string token"""
        start_line, start_column = self.line, self.column
        quote_char = self.current_char()
        result = ''
        self.advance()  # Skip opening quote
        
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == '\\':
                self.advance()
                if self.current_char():
                    # Handle escape sequences
                    escape_chars = {
                        'n': '\n', 't': '\t', 'r': '\r',
                        '\\': '\\', '"': '"', "'": "'"
                    }
                    result += escape_chars.get(self.current_char(), self.current_char())
                    self.advance()
            else:
                result += self.current_char()
                self.advance()
        
        if not self.current_char():
            self.error(f"Unterminated string starting at line {start_line}")
        else:
            self.advance()  # Skip closing quote
        
        return Token(TokenType.STRING, result, start_line, start_column)
    
    def read_identifier(self) -> Token:
        """Read identifier or keyword token"""
        start_line, start_column = self.line, self.column
        result = ''
        
        while (self.current_char() and 
               (self.current_char().isalnum() or self.current_char() == '_')):
            result += self.current_char()
            self.advance()
        
        token_type = TokenType.KEYWORD if result in self.KEYWORDS else TokenType.IDENTIFIER
        return Token(token_type, result, start_line, start_column)
    
    def read_comment(self) -> Token:
        """Read comment token"""
        start_line, start_column = self.line, self.column
        result = ''
        
        if self.current_char() == '/' and self.peek_char() == '/':
            # Single line comment
            while self.current_char() and self.current_char() != '\n':
                result += self.current_char()
                self.advance()
        elif self.current_char() == '/' and self.peek_char() == '*':
            # Multi-line comment
            self.advance()  # Skip /
            self.advance()  # Skip *
            result = '/*'
            
            while self.current_char():
                if self.current_char() == '*' and self.peek_char() == '/':
                    result += '*/'
                    self.advance()
                    self.advance()
                    break
                result += self.current_char()
                self.advance()
            else:
                self.error(f"Unterminated comment starting at line {start_line}")
        
        return Token(TokenType.COMMENT, result, start_line, start_column)
    
    def tokenize(self) -> List[Token]:
        """Tokenize the input text"""
        while self.current_char():
            self.skip_whitespace()
            
            if not self.current_char():
                break
            
            char = self.current_char()
            
            # Numbers
            if char.isdigit() or (char == '.' and self.peek_char() and self.peek_char().isdigit()):
                self.tokens.append(self.read_number())
            
            # Strings
            elif char in "\"'":
                self.tokens.append(self.read_string())
            
            # Identifiers and keywords
            elif char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
            
            # Comments
            elif char == '/' and self.peek_char() in '/*':
                self.tokens.append(self.read_comment())
            
            # Two-character operators
            elif char + (self.peek_char() or '') in self.OPERATORS:
                self.tokens.append(Token(TokenType.OPERATOR, 
                                      char + self.peek_char(), 
                                      self.line, self.column))
                self.advance()
                self.advance()
            
            # Single-character tokens
            elif char == '(':
                self.tokens.append(Token(TokenType.LPAREN, char, self.line, self.column))
                self.advance()
            elif char == ')':
                self.tokens.append(Token(TokenType.RPAREN, char, self.line, self.column))
                self.advance()
            elif char == '{':
                self.tokens.append(Token(TokenType.LBRACE, char, self.line, self.column))
                self.advance()
            elif char == '}':
                self.tokens.append(Token(TokenType.RBRACE, char, self.line, self.column))
                self.advance()
            elif char == '[':
                self.tokens.append(Token(TokenType.LBRACKET, char, self.line, self.column))
                self.advance()
            elif char == ']':
                self.tokens.append(Token(TokenType.RBRACKET, char, self.line, self.column))
                self.advance()
            elif char == ';':
                self.tokens.append(Token(TokenType.SEMICOLON, char, self.line, self.column))
                self.advance()
            elif char == ',':
                self.tokens.append(Token(TokenType.COMMA, char, self.line, self.column))
                self.advance()
            elif char == '=':
                self.tokens.append(Token(TokenType.EQUALS, char, self.line, self.column))
                self.advance()
            elif char == '\n':
                self.tokens.append(Token(TokenType.NEWLINE, char, self.line, self.column))
                self.advance()
            elif char in self.OPERATORS:
                self.tokens.append(Token(TokenType.OPERATOR, char, self.line, self.column))
                self.advance()
            else:
                self.error(f"Unexpected character: {char}")
                self.advance()
        
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens


def parse_script(script_text: str) -> Tuple[List[ASTNode], List[ParseError]]:
    """Parse script text and return AST and errors"""
    # Tokenize
    lexer = ScriptLexer(script_text)
    tokens = lexer.tokenize()
    
    # For now, return just the lexer results - parser implementation would continue here
    # This provides a foundation for the full parser implementation
    
    return [], lexer.errors
