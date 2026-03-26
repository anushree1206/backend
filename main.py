from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, UTC
import os
import ast
import re
import json
from groq import Groq
from dotenv import load_dotenv

# Optional MongoDB imports
try:
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoClient = None
    ObjectId = None

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Code Reviewer", version="2.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq configuration
api_key = os.getenv("GROQ_API_KEY")
if api_key == "gsk_your_api_key_here" or not api_key:
    print("WARNING: GROQ_API_KEY not set. AI features will be disabled.")
    groq_client = None
else:
    groq_client = Groq(api_key=api_key)

# MongoDB configuration (optional)
history_collection = None
mongodb_connected = False

if MONGODB_AVAILABLE:
    try:
        MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        db = client.code_reviewer
        history_collection = db.history
        mongodb_connected = True
        print("MongoDB connected successfully")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        print("Continuing without MongoDB - history features will be disabled")
        history_collection = None
        mongodb_connected = False
else:
    print("MongoDB not installed - history features will be disabled")

# ============= DATA MODELS =============
class AnalyzeRequest(BaseModel):
    code: str
    language: str

class AnalyzeResponse(BaseModel):
    original_code: str
    corrected_code: str
    errors: List[str]
    optimizations: List[str]
    summary: str
    scores: dict
    language: str
    created_at: str

class HistoryItem(BaseModel):
    id: str
    original_code: str
    corrected_code: str
    errors: List[str]
    optimizations: List[str]
    summary: str
    scores: dict
    language: str
    created_at: str

# ============= HELPER FUNCTIONS =============
def call_ai_api(prompt: str, system_message: str) -> Optional[str]:
    """Helper function to call AI API with error handling"""
    if not groq_client:
        return None
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"AI API call failed: {e}")
        return None

def parse_code_safely(code: str, language: str) -> tuple:
    """Parse code safely, return tree and errors"""
    if language.lower() == "python":
        try:
            tree = ast.parse(code)
            return tree, []
        except SyntaxError as e:
            return None, [f"Syntax Error at line {e.lineno}: {e.msg}"]
    return None, []

def detect_real_errors(code: str, language: str) -> List[str]:
    """Detect only real, major errors using AST"""
    errors = []
    
    if language.lower() == "python":
        tree, parse_errors = parse_code_safely(code, language)
        
        # Add syntax errors if any
        errors.extend(parse_errors)
        
        if tree:
            # Check for undefined variables (basic check)
            try:
                defined_vars = set()
                used_vars = set()
                
                for node in ast.walk(tree):
                    # Track variable definitions
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        defined_vars.add(node.id)
                    # Track variable usage
                    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        used_vars.add(node.id)
                
                # Check for undefined variables (excluding built-ins)
                builtins = {'print', 'len', 'range', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool'}
                undefined = used_vars - defined_vars - builtins
                for var in undefined:
                    errors.append(f"Undefined variable: '{var}'")
                    
            except Exception:
                pass  # Skip variable analysis if it fails
    
    return errors

def analyze_with_ai(code: str, language: str) -> dict:
    """Analyze code with strict AI prompts"""
    if not groq_client:
        return {
            "errors": [],
            "optimizations": [],
            "corrected_code": code,
            "summary": "Basic analysis: No AI available."
        }
    
    strict_prompt = f"""
STRICT RULES (MUST FOLLOW):

SUPPORTED LANGUAGES:
Python, C, C++, JavaScript, Java

LANGUAGE COMPLIANCE:
- Always follow the selected language strictly
- NEVER mix languages
- DO NOT use printf in C++
- DO NOT use cout in C

C++ REQUIREMENTS (if C++ selected):
- ALWAYS use #include <iostream>
- ALWAYS use cout
- ALWAYS use main() function

CODE CORRECTION RULE:
- If code is incomplete or incorrect:
  → Convert it into a COMPLETE working program in that language
- DO NOT return partial fixes
- DO NOT keep original incorrect style
- DO NOT ask for input again

Language: {language}

Code:
{code[:2000]}

OUTPUT FORMAT:
1. Errors (real only)
2. Optimizations (meaningful only)
3. Corrected Code (FULL PROGRAM ONLY)
4. Summary (1-2 lines)

If input language is C++ and code uses printf, you MUST convert it to cout.

Return JSON format:
{{
    "errors": ["error1", "error2"] or [],
    "optimizations": ["opt1", "opt2"] or [],
    "corrected_code": "complete corrected program",
    "summary": "1-2 line summary"
}}
"""
    
    system_message = "You are a strict code reviewer. Follow language rules exactly. Output only valid JSON."
    
    ai_response = call_ai_api(strict_prompt, system_message)
    
    if ai_response:
        try:
            # Try to parse JSON response
            if "```json" in ai_response:
                ai_response = ai_response.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_response:
                ai_response = ai_response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(ai_response)
            
            # Post-processing: Ensure C++ uses cout, not printf
            print(f"DEBUG: Language = '{language}'")
            print(f"DEBUG: Language lower = '{language.lower()}'")
            if "corrected_code" in result:
                print(f"DEBUG: Corrected code contains printf: {'printf' in result['corrected_code']}")
            
            if language.lower() in ["cpp", "c++"] and "corrected_code" in result:
                corrected_code = result["corrected_code"]
                print(f"DEBUG: Processing C++ code with printf: {'printf' in corrected_code}")

                if "printf" in corrected_code:
                    match = re.search(r'printf\("(.+?)"', corrected_code)
                    text = match.group(1) if match else "Output"
                    print(f"DEBUG: Extracted text: '{text}'")

                    corrected_code = f"""#include <iostream>
using namespace std;

int main() {{
    cout << "{text}";
    return 0;
}}"""
                    print(f"DEBUG: Converted to C++ code")

                result["corrected_code"] = corrected_code
            else:
                print(f"DEBUG: Not processing - language check failed")
            
            return result
        except json.JSONDecodeError:
            # Fallback: parse text response
            fallback_result = {
                "errors": [],
                "optimizations": [],
                "corrected_code": code,
                "summary": "Analysis completed."
            }
            
            # Post-processing for fallback case
            if language.lower() in ["cpp", "c++"] and "printf" in code:
                match = re.search(r'printf\("(.+?)"', code)
                text = match.group(1) if match else "Output"
                
                fallback_result["corrected_code"] = f"""#include <iostream>
using namespace std;

int main() {{
    cout << "{text}";
    return 0;
}}"""
                fallback_result["summary"] = "Converted printf to cout"
            
            return fallback_result
    
    # Fallback if AI fails
    fallback_result = {
        "errors": [],
        "optimizations": [],
        "corrected_code": code,
        "summary": "Analysis completed successfully."
    }
    
    # Final post-processing: Ensure C++ uses cout, not printf
    if language.lower() in ["cpp", "c++"] and "printf" in fallback_result["corrected_code"]:
        match = re.search(r'printf\("(.+?)"', fallback_result["corrected_code"])
        text = match.group(1) if match else "Output"
        
        fallback_result["corrected_code"] = f"""#include <iostream>
using namespace std;

int main() {{
    cout << "{text}";
    return 0;
}}"""
        fallback_result["summary"] = "Converted printf to cout"
    
    return fallback_result

def calculate_scores(code: str, errors: List[str], optimizations: List[str]) -> dict:
    """Calculate simplicity and readability scores"""
    lines = code.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Simplicity score (based on code complexity)
    simplicity_penalty = len(errors) * 15 + len(optimizations) * 5
    simplicity_score = max(0, 100 - simplicity_penalty - (len(non_empty_lines) * 0.5))
    
    # Readability score (based on various factors)
    readability_penalty = 0
    readability_penalty += len(errors) * 20
    readability_penalty += len(optimizations) * 10
    
    # Check for very long lines
    long_lines = sum(1 for line in lines if len(line) > 100)
    readability_penalty += long_lines * 5
    
    readability_score = max(0, 100 - readability_penalty)
    
    return {
        "simplicity": round(simplicity_score, 1),
        "readability": round(readability_score, 1)
    }

# ============= API ENDPOINTS =============
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_code(request: AnalyzeRequest):
    """Analyze code with AST and AI, store in MongoDB"""
    try:
        print(f"Received analysis request for {request.language} code")
        
        if not request.code.strip():
            raise HTTPException(status_code=400, detail="Code cannot be empty")
        
        # Step 1: Detect real errors with AST
        ast_errors = detect_real_errors(request.code, request.language)
        print(f"AST errors found: {len(ast_errors)}")
        
        # Step 2: AI analysis
        ai_result = analyze_with_ai(request.code, request.language)
        print(f"AI analysis completed")
        
        # Combine results
        all_errors = list(set(ast_errors + ai_result.get("errors", [])))
        optimizations = ai_result.get("optimizations", [])
        corrected_code = ai_result.get("corrected_code", request.code)
        summary = ai_result.get("summary", "Analysis completed.")
        
        # Calculate scores
        scores = calculate_scores(request.code, all_errors, optimizations)
        
        # Create result
        result = {
            "original_code": request.code,
            "corrected_code": corrected_code,
            "errors": all_errors[:10],  # Limit to 10
            "optimizations": optimizations[:10],  # Limit to 10
            "summary": summary,
            "scores": scores,
            "language": request.language,
            "created_at": datetime.now(UTC).isoformat()
        }
        
        print(f"Analysis result prepared successfully")
        
        # Store in MongoDB (optional)
        if history_collection is not None:
            try:
                history_collection.insert_one(result.copy())
                print("Stored in MongoDB")
            except Exception as e:
                print(f"Failed to store in MongoDB: {e}")
        else:
            print("MongoDB not available - skipping history storage")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/history", response_model=List[HistoryItem])
async def get_history():
    """Get all past submissions"""
    if history_collection is None:
        return []  # Return empty list if MongoDB not available
    
    try:
        history = []
        for doc in history_collection.find().sort("created_at", -1).limit(50):
            doc["id"] = str(doc.pop("_id"))
            history.append(HistoryItem(**doc))
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

@app.delete("/history/{id}")
async def delete_history_item(id: str):
    """Delete a specific history item"""
    if not MONGODB_AVAILABLE or history_collection is None:
        raise HTTPException(status_code=503, detail="History service unavailable")
    
    try:
        result = history_collection.delete_one({"_id": ObjectId(id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="History item not found")
        return {"message": "History item deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete history item: {str(e)}")

@app.delete("/history")
async def clear_history():
    """Clear all history"""
    if not MONGODB_AVAILABLE or history_collection is None:
        raise HTTPException(status_code=503, detail="History service unavailable")
    
    try:
        history_collection.delete_many({})
        return {"message": "History cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

# ============= HEALTH CHECK =============
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_available": groq_client is not None,
        "mongodb_connected": mongodb_connected
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting AI Code Reviewer Backend v2.0...")
    print("API Documentation: http://localhost:8001/docs")
    print("Health Check: http://localhost:8001/health")
    print("Tip: Configure GROQ_API_KEY and MONGODB_URI in .env")
    uvicorn.run(app, host="0.0.0.0", port=8001)
