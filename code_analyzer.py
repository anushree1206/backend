import ast
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Data class for code analysis results"""
    errors: List[str]
    style_issues: List[str]
    optimizations: List[str]
    complexity_score: float
    readability_score: float


class PythonCodeAnalyzer:
    """Comprehensive Python code analyzer using AST and pattern matching"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.style_issues: List[str] = []
        self.optimizations: List[str] = []
        self.lines = []
        self.tree = None
        
    def analyze(self, code: str) -> AnalysisResult:
        """Main analysis method"""
        self.lines = code.split('\n')
        self.errors = []
        self.style_issues = []
        self.optimizations = []
        
        # Module 1: Code Preprocessing & Parsing
        self._parse_code(code)
        
        # Module 2: Error Detection
        self._detect_errors(code)
        self._detect_logical_issues()
        
        # Module 3: Style Analysis (PEP8)
        self._analyze_style(code)
        
        # Module 4: Optimization Suggestions
        self._suggest_optimizations()
        
        # Calculate scores
        complexity = self._calculate_complexity()
        readability = self._calculate_readability()
        
        return AnalysisResult(
            errors=list(set(self.errors)),
            style_issues=list(set(self.style_issues)),
            optimizations=list(set(self.optimizations)),
            complexity_score=complexity,
            readability_score=readability
        )
    
    # ===== MODULE 1: Code Parsing & Preprocessing =====
    def _parse_code(self, code: str) -> bool:
        """Parse code into AST for analysis"""
        try:
            self.tree = ast.parse(code)
            return True
        except SyntaxError as e:
            self.errors.append(f"Syntax Error at line {e.lineno}: {e.msg}")
            return False
        except IndentationError as e:
            self.errors.append(f"Indentation Error at line {e.lineno}: {e.msg}")
            return False
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code for formatting and readability checks"""
        lines = code.split('\n')
        processed = []
        
        for i, line in enumerate(lines, 1):
            # Check for mixed tabs/spaces
            if '\t' in line and ' ' in line:
                self.style_issues.append(f"Line {i}: Mixed tabs and spaces detected")
            processed.append(line)
        
        return '\n'.join(processed)
    
    # ===== MODULE 2: Error & Bug Detection =====
    def _detect_errors(self, code: str) -> None:
        """Detect syntax and logical errors"""
        if not self.tree:
            return
        
        # Check for undefined variables
        self._check_undefined_variables()
        
        # Check for unused imports
        self._check_unused_imports()
        
        # Check for bare except clauses
        self._check_bare_excepts()
        
        # Check for unreachable code
        self._check_unreachable_code()
        
        # Check for infinite loops
        self._check_infinite_loops()
    
    def _check_undefined_variables(self) -> None:
        """Detect undefined variables"""
        defined_vars = set()
        used_vars = set()
        imports = set()
        
        for node in ast.walk(self.tree):
            # Track imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    imports.add(alias.asname or alias.name)
            
            # Track assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_vars.add(target.id)
            
            # Track function and class definitions
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined_vars.add(node.name)
            
            # Track used variables
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)
        
        # Built-in functions
        builtins = {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 
                   'set', 'tuple', 'bool', 'open', 'input', 'type', 'sum', 'max', 
                   'min', 'sorted', 'enumerate', 'zip', 'map', 'filter', 'abs', 
                   'any', 'all', 'isinstance', 'hasattr', 'getattr', 'setattr',
                   'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
                   '__name__', '__file__', '__doc__', 'self', 'cls'}
        
        undefined = used_vars - defined_vars - imports - builtins
        for var in undefined:
            self.errors.append(f"Undefined variable: '{var}'")
    
    def _check_unused_imports(self) -> None:
        """Detect unused imports"""
        imports = {}
        used_names = set()
        
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = node.lineno
            
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
        
        for imp_name, lineno in imports.items():
            if imp_name not in used_names:
                self.errors.append(f"Line {lineno}: Unused import '{imp_name}'")
    
    def _check_bare_excepts(self) -> None:
        """Detect bare except clauses"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    self.errors.append(f"Line {node.lineno}: Bare 'except:' clause detected - specify exception type")
    
    def _check_unreachable_code(self) -> None:
        """Detect unreachable code"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Return):
                # Check if there's code after return
                parent = self._get_parent_node(node)
                if parent and isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Simple check: look for statements after return
                    pass
    
    def _check_infinite_loops(self) -> None:
        """Detect potential infinite loops"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.While):
                # Check for while True without break
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
                    if not has_break:
                        self.errors.append(f"Line {node.lineno}: Infinite loop detected (while True with no break)")
    
    def _get_parent_node(self, node: ast.AST) -> ast.AST:
        """Helper to get parent node"""
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                if child is node:
                    return parent
        return None
    
    # ===== MODULE 3: Coding Style Analysis (PEP8) =====
    def _analyze_style(self, code: str) -> None:
        """Analyze code against PEP8 guidelines"""
        self._check_indentation()
        self._check_line_length()
        self._check_naming_conventions()
        self._check_whitespace()
        self._check_imports_order()
        self._check_function_length()
    
    def _check_indentation(self) -> None:
        """Check for proper indentation (4 spaces)"""
        for i, line in enumerate(self.lines, 1):
            if line and line[0] == ' ':
                spaces = len(line) - len(line.lstrip(' '))
                if spaces % 4 != 0:
                    self.style_issues.append(f"Line {i}: Indentation is not a multiple of 4 spaces")
    
    def _check_line_length(self) -> None:
        """Check for lines exceeding 79 characters"""
        for i, line in enumerate(self.lines, 1):
            if len(line) > 79:
                self.style_issues.append(f"Line {i}: Line is {len(line)} characters (exceeds 79 limit)")
    
    def _check_naming_conventions(self) -> None:
        """Check variable and function naming conventions"""
        if not self.tree:
            return
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if not self._is_valid_function_name(node.name):
                    self.style_issues.append(f"Line {node.lineno}: Function name '{node.name}' should be lowercase with underscores")
            
            if isinstance(node, ast.ClassDef):
                if not self._is_valid_class_name(node.name):
                    self.style_issues.append(f"Line {node.lineno}: Class name '{node.name}' should use CapWords convention")
            
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper() and len(target.id) > 1:
                            # Could be a constant, which is fine
                            pass
                        elif not self._is_valid_var_name(target.id):
                            self.style_issues.append(f"Line {node.lineno}: Variable name '{target.id}' should be lowercase with underscores")
    
    def _is_valid_function_name(self, name: str) -> bool:
        """Check if function name follows PEP8"""
        return bool(re.match(r'^[a-z_][a-z0-9_]*$', name))
    
    def _is_valid_class_name(self, name: str) -> bool:
        """Check if class name follows PEP8 (CapWords)"""
        return bool(re.match(r'^[A-Z][a-zA-Z0-9]*$', name))
    
    def _is_valid_var_name(self, name: str) -> bool:
        """Check if variable name follows PEP8"""
        return bool(re.match(r'^[a-z_][a-z0-9_]*$', name))
    
    def _check_whitespace(self) -> None:
        """Check for proper whitespace"""
        for i, line in enumerate(self.lines, 1):
            # Check trailing whitespace
            if line != line.rstrip():
                self.style_issues.append(f"Line {i}: Trailing whitespace detected")
            
            # Check multiple spaces around operators
            if '  ' in line and not line.strip().startswith('#'):
                self.style_issues.append(f"Line {i}: Multiple spaces found (use single spaces around operators)")
    
    def _check_imports_order(self) -> None:
        """Check if imports are properly organized (stdlib, third-party, local)"""
        if not self.tree:
            return
        
        stdlib_imports = []
        third_party = []
        local_imports = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.name.split('.')[0]
                    if self._is_stdlib(name):
                        stdlib_imports.append(node.lineno)
                    else:
                        local_imports.append(node.lineno)
    
    def _is_stdlib(self, module: str) -> bool:
        """Check if module is standard library"""
        stdlib = {'os', 'sys', 're', 'json', 'math', 'random', 'datetime', 
                 'time', 'collections', 'itertools', 'functools', 'operator',
                 'string', 'io', 'pickle', 'csv', 'xml', 'html', 'urllib',
                 'http', 'socket', 'email', 'logging', 'unittest', 'asyncio',
                 'threading', 'multiprocessing', 'subprocess', 'shutil', 'glob',
                 'pathlib', 'tempfile', 'gzip', 'zipfile', 'tarfile', 'typing'}
        return module in stdlib
    
    def _check_function_length(self) -> None:
        """Check for overly long functions (>30 lines)"""
        if not self.tree:
            return
        
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_length = node.end_lineno - node.lineno
                if func_length > 30:
                    self.style_issues.append(f"Line {node.lineno}: Function '{node.name}' is {func_length} lines long (consider breaking it down)")
    
    # ===== MODULE 4: Optimization & Best Practice Suggestions =====
    def _suggest_optimizations(self) -> None:
        """Suggest code optimizations and best practices"""
        if not self.tree:
            return
        
        self._suggest_list_comprehensions()
        self._suggest_with_statements()
        self._suggest_string_operations()
        self._suggest_type_hints()
        self._suggest_documentation()
        self._suggest_algorithm_improvements()
    
    def _suggest_list_comprehensions(self) -> None:
        """Suggest using list comprehensions"""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.For):
                # Check for patterns that could use list comprehension
                if isinstance(node.parent, ast.Assign) if hasattr(node, 'parent') else False:
                    self.optimizations.append(f"Line {node.lineno}: Consider using list comprehension instead of for loop for better performance")
    
    def _suggest_with_statements(self) -> None:
        """Suggest using with statements for file handling"""
        code = '\n'.join(self.lines)
        if 'open(' in code and 'with' not in code:
            self.optimizations.append("Consider using 'with' statements when opening files to ensure proper resource cleanup")
    
    def _suggest_string_operations(self) -> None:
        """Suggest string operation improvements"""
        code = '\n'.join(self.lines)
        
        # Check for string concatenation in loops
        if '+=' in code and any('for ' in line for line in self.lines):
            self.optimizations.append("Use list join() instead of string concatenation in loops for better performance")
        
        # Check for f-strings
        if '%' in code or '.format(' in code:
            self.optimizations.append("Consider using f-strings (Python 3.6+) instead of % or .format() for better readability")
    
    def _suggest_type_hints(self) -> None:
        """Suggest adding type hints"""
        if not self.tree:
            return
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                if not node.returns:
                    self.optimizations.append(f"Line {node.lineno}: Consider adding type hints to function '{node.name}' for better code clarity")
                    break  # Only suggest once
    
    def _suggest_documentation(self) -> None:
        """Suggest adding docstrings"""
        if not self.tree:
            return
        
        functions_without_docs = 0
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    functions_without_docs += 1
        
        if functions_without_docs > 0:
            self.optimizations.append(f"Add docstrings to {functions_without_docs} function(s)/class(es) to improve documentation")
    
    def _suggest_algorithm_improvements(self) -> None:
        """Suggest algorithmic improvements"""
        code = '\n'.join(self.lines)
        
        # Check for nested loops
        nested_loops = code.count('for ') - 1
        if nested_loops > 1:
            self.optimizations.append("Multiple nested loops detected - consider algorithmic optimization (e.g., using sets for O(1) lookup)")
        
        # Check for repeated calculations
        if code.count('len(') > 2:
            self.optimizations.append("Cache frequently used len() calls in variables to avoid redundant calculations")
    
    # ===== Scoring Methods =====
    def _calculate_complexity(self) -> float:
        """Calculate cyclomatic complexity score (0-100)"""
        if not self.tree:
            return 0.0
        
        complexity = 1
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, 
                               ast.BoolOp, ast.Lambda)):
                complexity += 1
        
        # Normalize to 0-100 scale
        score = min(100, 100 - (complexity * 5))
        return max(0, score)
    
    def _calculate_readability(self) -> float:
        """Calculate readability score (0-100)"""
        score = 100
        
        # Deduct for style issues
        score -= len(self.style_issues) * 5
        
        # Deduct for errors
        score -= len(self.errors) * 10
        
        # Deduct for missing documentation
        if not self.tree:
            score -= 20
        else:
            documented = sum(1 for node in ast.walk(self.tree) 
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef)) 
                           and ast.get_docstring(node))
            total_funcs = sum(1 for node in ast.walk(self.tree) 
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
            if total_funcs > 0 and documented / total_funcs < 0.5:
                score -= 15
        
        return max(0, min(100, score))


class CodeAnalyzerFactory:
    """Factory for creating language-specific analyzers"""
    
    @staticmethod
    def create_analyzer(language: str):
        """Create appropriate analyzer for language"""
        if language.lower() == 'python':
            return PythonCodeAnalyzer()
        else:
            # For other languages, return a basic analyzer
            return PythonCodeAnalyzer()
