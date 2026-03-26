from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
import ast
import re
import json
import builtins
from groq import Groq

app = FastAPI(title="Code Review API", version="1.0.0")

# CORS configuration - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq configuration
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============= DATA MODELS =============
class CodeReviewRequest(BaseModel):
    code: str
    language: str
    filename: Optional[str] = None


class CodeReviewResponse(BaseModel):
    id: str
    code: str
    language: str
    filename: Optional[str]
    errors: List[str]
    style_issues: List[str]
    optimizations: List[str]
    complexity_score: float
    readability_score: float
    created_at: str


# ============= MODULE 1: CODE PARSING & PREPROCESSING =============
def parse_code(code: str, language: str) -> tuple:
    """Parse code using AST for Python with enhanced error handling"""
    if language.lower() == "python":
        try:
            tree = ast.parse(code)
            normalized_code = ast.unparse(tree) if hasattr(ast, 'unparse') else code
            return tree, []
        except SyntaxError as e:
            error_msg = f"Syntax Error at line {e.lineno}: {e.msg}"
            return None, [error_msg]
        except Exception as e:
            return None, [f"Parsing error: {str(e)}"]
    else:
        # For non-Python languages, return None and empty errors
        return None, []


# ============= ENHANCED AST ERROR DETECTOR =============
class ErrorVisitor(ast.NodeVisitor):
    def __init__(self):
        self.assigned_vars = set()
        self.used_vars = set()
        self.imported_modules = set()
        self.function_params = set()
        self.errors = []
        self.function_lengths = []
        self.naming_violations = []
        self.style_score = 100

    # -------------------------
    # FUNCTION CHECKS
    # -------------------------
    def visit_FunctionDef(self, node):
        # Collect parameters
        for arg in node.args.args:
            self.function_params.add(arg.arg)

        # Check snake_case
        if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
            self.naming_violations.append({
                "type": "Function Naming",
                "line": node.lineno,
                "message": f"Function '{node.name}' should be snake_case."
            })
            self.style_score -= 10

        # Check function length
        if hasattr(node, "end_lineno"):
            length = node.end_lineno - node.lineno
            if length > 40:
                self.function_lengths.append({
                    "line": node.lineno,
                    "message": f"Function '{node.name}' is too long ({length} lines)."
                })
                self.style_score -= 5

        # Unreachable code detection
        found_return = False
        for stmt in node.body:
            if found_return:
                self.errors.append({
                    "category": "Logical Error",
                    "line": stmt.lineno,
                    "error": "Unreachable Code",
                    "message": "Code after return statement will never execute.",
                    "explanation": "Any statement written after return inside a function is unreachable."
                })
            if isinstance(stmt, ast.Return):
                found_return = True

        self.generic_visit(node)

    # -------------------------
    # CLASS CHECK
    # -------------------------
    def visit_ClassDef(self, node):
        # PascalCase check
        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
            self.naming_violations.append({
                "type": "Class Naming",
                "line": node.lineno,
                "message": f"Class '{node.name}' should be PascalCase."
            })
            self.style_score -= 10

        self.generic_visit(node)

    # -------------------------
    # VARIABLE TRACKING
    # -------------------------
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned_vars.add(target.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_vars.add(node.id)
        self.generic_visit(node)

    # -------------------------
    # IMPORTS
    # -------------------------
    def visit_Import(self, node):
        for alias in node.names:
            self.imported_modules.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imported_modules.add(alias.asname or alias.name)
        self.generic_visit(node)

    # -------------------------
    # INFINITE LOOP
    # -------------------------
    def visit_While(self, node):
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
            if not has_break:
                self.errors.append({
                    "category": "Logical Error",
                    "line": node.lineno,
                    "error": "Infinite Loop",
                    "message": "This while loop runs forever.",
                    "explanation": "The loop condition is always True and no break statement is found."
                })
        self.generic_visit(node)


def detect_errors(tree):
    visitor = ErrorVisitor()
    visitor.visit(tree)

    errors = visitor.errors

    # Built-in names
    builtin_names = set(dir(__builtins__))

    # Undefined variables
    undefined = (
        visitor.used_vars
        - visitor.assigned_vars
        - visitor.imported_modules
        - visitor.function_params
        - builtin_names
    )

    for var in undefined:
        errors.append({
            "category": "Logical Error",
            "line": None,
            "error": "Undefined Variable",
            "message": f"Variable '{var}' used before assignment.",
            "explanation": "Variables must be defined before they are used."
        })

    # Unused imports
    unused_imports = visitor.imported_modules - visitor.used_vars
    for module in unused_imports:
        errors.append({
            "category": "Logical Issue",
            "line": None,
            "error": "Unused Import",
            "message": f"Module '{module}' is imported but never used.",
            "explanation": "Remove unused imports to keep code clean."
        })

    return {
        "errors": errors,
        "naming_violations": visitor.naming_violations,
        "long_functions": visitor.function_lengths,
        "style_score": max(visitor.style_score, 0)
    }
            tree = ast.parse(code)
            return tree, []
        except SyntaxError as e:
            return None, [f"Syntax Error at line {e.lineno}: {e.msg}"]
        except IndentationError as e:
            return None, [f"Indentation Error at line {e.lineno}: {e.msg}"]
    return None, []


# ============= MODULE 2: ERROR & BUG DETECTION =============
def detect_errors(code: str, language: str) -> List[str]:
    """Detect errors using enhanced AST analysis and AI"""
    errors = []
    
    if language.lower() == "python":
        # Parse the code first
        tree, parse_errors = parse_code(code, language)
        errors.extend(parse_errors)
        
        if tree:
            # Use AST-based error detection
            ast_analysis = detect_errors(tree)
            
            # Convert AST errors to string format
            for error in ast_analysis["errors"]:
                if isinstance(error, dict):
                    error_msg = f"{error.get('error', 'Error')}: {error.get('message', 'Unknown error')}"
                    if error.get('line'):
                        error_msg += f" (Line {error['line']})"
                    errors.append(error_msg)
                else:
                    errors.append(str(error))
            
            # Add naming violations as style issues
            for violation in ast_analysis["naming_violations"]:
                if isinstance(violation, dict):
                    error_msg = f"{violation.get('type', 'Naming Issue')}: {violation.get('message', 'Naming problem')}"
                    if violation.get('line'):
                        error_msg += f" (Line {violation['line']})"
                    errors.append(error_msg)
            
            # Add long function warnings
            for func_info in ast_analysis["long_functions"]:
                if isinstance(func_info, dict):
                    error_msg = f"Long Function: {func_info.get('message', 'Function is too long')}"
                    if func_info.get('line'):
                        error_msg += f" (Line {func_info['line']})"
                    errors.append(error_msg)
    
    # AI-powered error detection for comprehensive analysis
    try:
        prompt = f"""
        Analyze the following {language} code for errors, bugs, and potential issues:
        
        ```{language}
        {code}
        ```
        
        IMPORTANT: Provide ONLY specific, actionable issues. Focus on:
        1. Syntax errors and undefined variables
        2. Logic errors that would cause runtime failures
        3. Security vulnerabilities
        4. Performance issues that would cause problems
        5. Critical bugs that break functionality
        
        Format each issue as a clear, concise statement starting with what's wrong.
        Example: "Variable 'x' is used before being defined on line 5"
        Example: "Function 'calculate' has unreachable code after return statement"
        
        Do NOT include general advice or suggestions - only actual errors.
        """
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert code reviewer identifying errors and bugs. Be thorough and specific."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.2
        )
        
        ai_errors = response.choices[0].message.content.strip()
        # Parse AI response into list items
        for line in ai_errors.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                errors.append(line.lstrip('-*• ').strip())
            elif line and not line.startswith('```') and not line.startswith('#'):
                errors.append(line)
                
    except Exception as e:
        print(f"AI error analysis failed: {e}")
    
    # Add some local heuristic checks as fallback
    if language.lower() == "python":
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if 'print(' in line and not line.strip().endswith(')'):
                errors.append(f"Possible syntax error on line {i}: incomplete print statement")
            if 'import' in line and line.strip().endswith(','):
                errors.append(f"Possible syntax error on line {i}: incomplete import statement")
    
    return list(set(errors))[:15]  # Remove duplicates and limit to 15 errors


# ============= MODULE 3: STYLE ANALYSIS (PEP8) =============
def analyze_style(code: str, language: str) -> List[str]:
    """Analyze code style using AST and AI"""
    style_issues = []
    
    if language.lower() == "python":
        # Parse the code for AST analysis
        tree, parse_errors = parse_code(code, language)
        
        if tree:
            # Use the AST visitor for style analysis
            ast_analysis = detect_errors(tree)
            
            # Add naming violations from AST analysis
            for violation in ast_analysis["naming_violations"]:
                if isinstance(violation, dict):
                    issue_msg = f"{violation.get('type', 'Naming Issue')}: {violation.get('message', 'Naming problem')}"
                    if violation.get('line'):
                        issue_msg += f" (Line {violation['line']})"
                    style_issues.append(issue_msg)
            
            # Add long function warnings
            for func_info in ast_analysis["long_functions"]:
                if isinstance(func_info, dict):
                    issue_msg = f"Long Function: {func_info.get('message', 'Function is too long')}"
                    if func_info.get('line'):
                        issue_msg += f" (Line {func_info['line']})"
                    style_issues.append(issue_msg)
    
    # AI-powered style analysis for comprehensive checking
    try:
        prompt = f"""
        Analyze the following {language} code for style issues and best practices:
        
        ```{language}
        {code}
        ```
        
        IMPORTANT: Focus on actionable style improvements:
        1. Naming convention violations (snake_case, PascalCase, etc.)
        2. Code formatting issues (indentation, spacing, line length)
        3. Missing documentation or comments
        4. Code organization and structure problems
        5. Language-specific style guideline violations
        
        Format each issue as a clear, actionable recommendation.
        Example: "Function 'calculateSum' should use snake_case naming"
        Example: "Line 15 exceeds 79 character limit (85 characters)"
        Example: "Add docstring to function 'process_data'"
        
        Be specific and provide line numbers when possible.
        """
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert code reviewer focusing on style and best practices. Be constructive and specific."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.3
        )
        
        ai_style_issues = response.choices[0].message.content.strip()
        # Parse AI response into list items
        for line in ai_style_issues.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                style_issues.append(line.lstrip('-*• ').strip())
            elif line and not line.startswith('```') and not line.startswith('#'):
                style_issues.append(line)
                
    except Exception as e:
        print(f"AI style analysis failed: {e}")
    
    # Add local style checks as fallback
    if language.lower() == "python":
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 79:
                style_issues.append(f"Line {i} exceeds 79 characters ({len(line)} characters)")
            # Check trailing whitespace
            if line.rstrip() != line:
                style_issues.append(f"Line {i}: Trailing whitespace")
            # Check for multiple blank lines
            if i > 1 and not lines[i-2].strip() and not lines[i-1].strip() and not line.strip():
                style_issues.append(f"Line {i}: Too many blank lines")
    
    return list(set(style_issues))[:12]  # Remove duplicates and limit to 12 issues


# ============= MODULE 4: OPTIMIZATION SUGGESTIONS =============
def get_optimization_suggestions(code: str, language: str) -> List[str]:
    """Get optimization suggestions using Groq AI"""
    suggestions = []
    
    try:
        prompt = f"""
        Analyze the following {language} code and provide specific optimization suggestions:
        
        ```{language}
        {code}
        ```
        
        IMPORTANT: Provide ONLY actionable optimization opportunities:
        1. Performance improvements (algorithms, data structures)
        2. Code simplification and readability
        3. Memory usage optimization
        4. Modern language features or best practices
        5. Elimination of redundant or inefficient code
        
        Format each suggestion as a clear, actionable recommendation.
        Example: "Replace manual loop with list comprehension for better performance"
        Example: "Use 'with open()' instead of manual file handling"
        Example: "Cache the result of expensive function call 'calculate_data'"
        
        Focus on changes that will have measurable impact.
        """
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert software engineer providing code optimization advice. Be specific and practical."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        ai_suggestions = response.choices[0].message.content.strip()
        # Parse AI response into list items
        for line in ai_suggestions.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                suggestions.append(line.lstrip('-*• ').strip())
        
        # Add some local optimizations as fallback
        if language.lower() == "python":
            lines = code.split('\n')
            if 'for ' in code and '[' in code:
                if 'append' in code:
                    suggestions.append("Consider using list comprehension instead of append() in loops")
            if ' if ' in code and ' else ' in code:
                if 'True' in code or 'False' in code:
                    suggestions.append("Simplify conditional logic where possible")
            if len(lines) > 10:
                suggestions.append("Consider extracting repeated code into functions")
                
    except Exception as e:
        print(f"AI optimization analysis failed: {e}")
        # Fallback to local suggestions
        if language.lower() == "python":
            suggestions.append("Consider using list comprehensions for better performance")
            suggestions.append("Review variable naming for clarity")
            suggestions.append("Look for opportunities to reduce code duplication")
    
    return suggestions[:10]  # Limit to 10 suggestions


def generate_optimized_code(code: str, language: str) -> str:
    """Generate optimized version of the code using AI"""
    try:
        prompt = f"""
        Optimize the following {language} code for better performance, readability, and modern practices:
        
        ```{language}
        {code}
        ```
        
        Requirements:
        1. Fix any performance issues
        2. Improve code structure and readability
        3. Use modern language features
        4. Maintain the same functionality
        5. Add comments explaining key optimizations
        
        Return ONLY the optimized code without explanations.
        """
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert software engineer who optimizes code for performance and readability. Return only the optimized code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        
        optimized_code = response.choices[0].message.content.strip()
        
        # Clean up the response - remove any markdown code blocks
        if optimized_code.startswith('```'):
            lines = optimized_code.split('\n')
            if lines[0].startswith('```'):
                optimized_code = '\n'.join(lines[1:-1]) if lines[-1].startswith('```') else '\n'.join(lines[1:])
        
        return optimized_code
        
    except Exception as e:
        print(f"AI code optimization failed: {e}")
        return code  # Return original code if optimization fails


# ============= CALCULATE SCORES =============
def calculate_complexity_score(code: str, language: str) -> float:
    """Calculate code complexity score (0-100)"""
    if language.lower() != "python":
        return 50.0
    
    tree, _ = parse_code(code, language)
    if not tree:
        return 0.0
    
    complexity = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
            complexity += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity += 2
        elif isinstance(node, ast.ClassDef):
            complexity += 3
        elif isinstance(node, ast.ListComp):
            complexity += 1
        elif isinstance(node, ast.DictComp):
            complexity += 1
    
    # Normalize to 0-100 scale (lower complexity is better)
    return min(complexity * 2, 100)


def calculate_readability_score(code: str, language: str, errors: List[str], style_issues: List[str]) -> float:
    """Calculate readability score (0-100)"""
    base_score = 100
    
    # Deduct points for errors
    base_score -= len(errors) * 10
    
    # Deduct points for style issues
    base_score -= len(style_issues) * 5
    
    # Additional checks for Python
    if language.lower() == "python":
        lines = code.split('\n')
        # Check for very long lines
        long_lines = sum(1 for line in lines if len(line) > 100)
        base_score -= long_lines * 2
        
        # Check for very short variable names
        tree, _ = parse_code(code, language)
        if tree:
            short_names = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if len(node.id) == 1 and node.id not in ['i', 'j', 'k', 'x', 'y', 'z']:
                        short_names += 1
            base_score -= short_names * 3
    
    return max(base_score, 0)


# ============= MAIN REVIEW ENDPOINT =============
@app.post("/api/review")
async def review_code(request: CodeReviewRequest):
    """
    MODULE 5: Submit code for comprehensive review
    Performs all analyses: parsing, error detection, style check, optimization
    """
    
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    
    # Module 1: Parse code
    tree, parse_errors = parse_code(request.code, request.language)
    
    # Module 2: Detect errors
    errors = detect_errors(request.code, request.language)
    
    # Module 3: Style analysis
    style_issues = analyze_style(request.code, request.language)
    
    # Module 4: Optimizations
    optimizations = get_optimization_suggestions(request.code, request.language)
    
    # Calculate scores
    complexity_score = calculate_complexity_score(request.code, request.language)
    readability_score = calculate_readability_score(request.code, request.language, errors, style_issues)
    
    # Generate unique ID
    review_id = f"review_{int(datetime.utcnow().timestamp() * 1000)}"
    
    return {
        "id": review_id,
        "code": request.code,
        "language": request.language,
        "filename": request.filename,
        "errors": list(set(errors))[:10],  # Limit to 10
        "style_issues": list(set(style_issues))[:10],
        "optimizations": list(set(optimizations))[:10],
        "complexity_score": round(complexity_score, 2),
        "readability_score": round(readability_score, 2),
        "created_at": datetime.utcnow().isoformat()
    }


# ============= OPTIMIZED CODE ENDPOINT =============
@app.post("/api/optimize")
async def optimize_code(request: CodeReviewRequest):
    """
    Generate optimized version of the submitted code
    """
    
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    
    # Generate optimized code
    optimized_code = generate_optimized_code(request.code, request.language)
    
    # Also provide optimization suggestions
    optimizations = get_optimization_suggestions(request.code, request.language)
    
    return {
        "original_code": request.code,
        "optimized_code": optimized_code,
        "language": request.language,
        "filename": request.filename,
        "optimizations": list(set(optimizations))[:10],
        "created_at": datetime.utcnow().isoformat()
    }


# ============= HISTORY ENDPOINTS =============
@app.get("/api/reviews")
async def get_reviews(skip: int = 0, limit: int = 10):
    """Get review history (mock implementation)"""
    return {
        "reviews": [],
        "total": 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
        # Fallback to local suggestions
        if language.lower() == "python":
            suggestions.append("Consider using list comprehensions for better performance")
            suggestions.append("Review variable naming for clarity")
            suggestions.append("Look for opportunities to reduce code duplication")
    
    return suggestions[:10]  # Limit to 10 suggestions


def generate_optimized_code(code: str, language: str) -> str:
    """Generate optimized version of the code using AI"""
    try:
        prompt = f"""
        Optimize the following {language} code for better performance, readability, and modern practices:
        
        ```{language}
        {code}
        ```
        
        Requirements:
        1. Fix any performance issues
        2. Improve code structure and readability
        3. Use modern language features
        4. Maintain the same functionality
        5. Add comments explaining key optimizations
        
        Return ONLY the optimized code without explanations.
        """
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert software engineer who optimizes code for performance and readability. Return only the optimized code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.2
        )
        
        optimized_code = response.choices[0].message.content.strip()
        
        # Clean up the response - remove any markdown code blocks
        if optimized_code.startswith('```'):
            lines = optimized_code.split('\n')
            if lines[0].startswith('```'):
                optimized_code = '\n'.join(lines[1:-1]) if lines[-1].startswith('```') else '\n'.join(lines[1:])
        
        return optimized_code
        
    except Exception as e:
        print(f"AI code optimization failed: {e}")
        return code  # Return original code if optimization fails


# ============= CALCULATE SCORES =============
def calculate_complexity_score(code: str, language: str) -> float:
    """Calculate code complexity score (0-100)"""
    if language.lower() != "python":
        return 50.0
    
    tree, _ = parse_code(code, language)
    if not tree:
        return 0.0
    
    # Simple complexity calculation based on nesting depth
    max_depth = 0
    
    class DepthVisitor(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0
            self.max_depth = 0
        
        def visit(self, node):
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            self.generic_visit(node)
            self.depth -= 1
    
    visitor = DepthVisitor()
    visitor.visit(tree)
    
    # Scale: depth 0-3 = simple, 4-6 = moderate, 7+ = complex
    complexity = min(100, (visitor.max_depth / 10) * 100)
    return 100 - complexity  # Invert so lower complexity = higher score


def calculate_readability_score(code: str, language: str, errors: List[str], style_issues: List[str]) -> float:
    """Calculate readability score based on errors and style issues"""
    base_score = 100.0
    
    # Deduct points for errors
    base_score -= len(errors) * 15
    
    # Deduct points for style issues
    base_score -= len(style_issues) * 5
    
    # Check for meaningful variable names
    if language.lower() == "python":
        tree, _ = parse_code(code, language)
        if tree:
            bad_names = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if len(node.id) == 1 and node.id not in ('i', 'j', 'k', 'x', 'y', 'z'):
                        bad_names += 1
            base_score -= (bad_names * 2)
    
    return max(0, min(100, base_score))


# ============= MAIN REVIEW ENDPOINT =============
@app.post("/api/review")
async def review_code(request: CodeReviewRequest):
    """
    MODULE 5: Submit code for comprehensive review
    Performs all analyses: parsing, error detection, style check, optimization
    """
    
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    
    # Module 1: Parse code
    tree, parse_errors = parse_code(request.code, request.language)
    
    # Module 2: Detect errors
    errors = detect_errors(request.code, request.language)
    
    # Module 3: Style analysis
    style_issues = analyze_style(request.code, request.language)
    
    # Module 4: Optimizations
    optimizations = get_optimization_suggestions(request.code, request.language)
    
    # Calculate scores
    complexity_score = calculate_complexity_score(request.code, request.language)
    readability_score = calculate_readability_score(request.code, request.language, errors, style_issues)
    
    # Generate unique ID
    review_id = f"review_{int(datetime.utcnow().timestamp() * 1000)}"
    
    return {
        "id": review_id,
        "code": request.code,
        "language": request.language,
        "filename": request.filename,
        "errors": list(set(errors))[:10],  # Limit to 10
        "style_issues": list(set(style_issues))[:10],
        "optimizations": list(set(optimizations))[:10],
        "complexity_score": round(complexity_score, 2),
        "readability_score": round(readability_score, 2),
        "created_at": datetime.utcnow().isoformat()
    }


# ============= OPTIMIZED CODE ENDPOINT =============
@app.post("/api/optimize")
async def optimize_code(request: CodeReviewRequest):
    """
    Generate optimized version of the submitted code
    """
    
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    
    # Generate optimized code
    optimized_code = generate_optimized_code(request.code, request.language)
    
    # Also provide optimization suggestions
    optimizations = get_optimization_suggestions(request.code, request.language)
    
    return {
        "original_code": request.code,
        "optimized_code": optimized_code,
        "language": request.language,
        "filename": request.filename,
        "optimizations": list(set(optimizations))[:10],
        "created_at": datetime.utcnow().isoformat()
    }


# ============= HISTORY ENDPOINTS =============
@app.get("/api/reviews")
async def get_reviews(skip: int = 0, limit: int = 10):
    """Get review history (mock implementation)"""
    return {
        "reviews": [],
        "total": 0
    }


@app.get("/api/reviews/{review_id}")
async def get_review_detail(review_id: str):
    """Get review details (mock implementation)"""
    raise HTTPException(status_code=404, detail="Review not found")


# ============= HEALTH CHECK =============
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Code Review API is running"}


@app.get("/docs")
async def docs():
    """API documentation"""
    return {"docs": "Available at /docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
