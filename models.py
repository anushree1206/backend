from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId to handle MongoDB IDs"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class CodeReviewBase(BaseModel):
    """Base model for code reviews"""
    code: str
    language: str
    filename: Optional[str] = None
    errors: List[str] = []
    style_issues: List[str] = []
    optimizations: List[str] = []


class CodeReviewCreate(CodeReviewBase):
    """Model for creating code reviews"""
    pass


class CodeReviewInDB(CodeReviewBase):
    """Model for code reviews in database"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "code": "print('hello')",
                "language": "python",
                "filename": "hello.py",
                "errors": [],
                "style_issues": [],
                "optimizations": ["Use f-strings for string formatting"],
                "created_at": "2026-03-13T12:00:00"
            }
        }


class ReviewFeedback(BaseModel):
    """Model for review feedback from Groq"""
    errors: List[str]
    style_issues: List[str]
    optimizations: List[str]


class CodeReviewResponse(BaseModel):
    """Model for API response"""
    id: str
    code: str
    language: str
    filename: Optional[str]
    errors: List[str]
    style_issues: List[str]
    optimizations: List[str]
    created_at: str


class ReviewsListResponse(BaseModel):
    """Model for list of reviews response"""
    reviews: List[dict]
    total: int


# Database Collections Configuration
class CollectionConfig:
    """Configuration for MongoDB collections"""
    
    CODE_REVIEWS_COLLECTION = "code_reviews"
    
    # Index definitions for performance
    INDEXES = {
        CODE_REVIEWS_COLLECTION: [
            ("created_at", -1),  # Sort by creation date
            ("language", 1),     # Filter by language
            ("filename", 1),     # Search by filename
        ]
    }
