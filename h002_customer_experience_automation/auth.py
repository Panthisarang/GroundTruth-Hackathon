import bcrypt
import re
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import logging

from h002_customer_experience_automation.config import SECRET_KEY
from h002_customer_experience_automation.db import db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthError(Exception):
    """Base authentication error class."""
    pass

class InvalidEmailError(AuthError):
    """Raised when an invalid email is provided."""
    pass

class WeakPasswordError(AuthError):
    """Raised when the password doesn't meet requirements."""
    pass

class UserExistsError(AuthError):
    """Raised when trying to create a user that already exists."""
    pass

class InvalidCredentialsError(AuthError):
    """Raised when login credentials are invalid."""
    pass

class AuthService:
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """Validate password strength.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        if not re.search(r'[0-9]', password):
            return False, "Password must contain at least one number"
        if not re.search(r'[^A-Za-z0-9]', password):
            return False, "Password must contain at least one special character"
        return True, ""
    
    @staticmethod
    def hash_password(password: str) -> bytes:
        """Hash a password using bcrypt."""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    @staticmethod
    def check_password(password: str, hashed: bytes) -> bool:
        """Check if a password matches the hashed version."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed)
        except Exception as e:
            logger.error(f"Error checking password: {e}")
            return False
    
    @classmethod
    def create_user(cls, email: str, password: str, **user_data) -> str:
        """Create a new user with the given credentials and data.
        
        Args:
            email: User's email address
            password: Plain text password
            **user_data: Additional user data (name, preferences, etc.)
            
        Returns:
            The ID of the created user
            
        Raises:
            InvalidEmailError: If the email is invalid
            WeakPasswordError: If the password is too weak
            UserExistsError: If a user with the email already exists
        """
        # Validate email
        if not cls.validate_email(email):
            raise InvalidEmailError("Invalid email format")
        
        # Validate password
        is_valid, error = cls.validate_password(password)
        if not is_valid:
            raise WeakPasswordError(error)
        
        # Check if user already exists
        if db.get_user_by_email(email):
            raise UserExistsError("A user with this email already exists")
        
        # Hash password
        hashed_password = cls.hash_password(password)
        
        # Prepare user document
        user_doc = {
            "email": email,
            "password": hashed_password,
            "created_at": datetime.utcnow(),
            "last_login": None,
            "llm_usage_count": 0,
            **user_data
        }
        
        # Store user in database
        user_id = db.create_user(user_doc)
        logger.info(f"Created new user: {email}")
        
        return user_id
    
    @classmethod
    def verify_user(cls, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials.
        
        Args:
            email: User's email
            password: Plain text password
            
        Returns:
            User document if credentials are valid, None otherwise
            
        Raises:
            InvalidCredentialsError: If credentials are invalid
        """
        user = db.get_user_by_email(email)
        if not user:
            logger.warning(f"Login attempt with non-existent email: {email}")
            raise InvalidCredentialsError("Invalid email or password")
        
        if not cls.check_password(password, user["password"]):
            logger.warning(f"Invalid password attempt for user: {email}")
            raise InvalidCredentialsError("Invalid email or password")
        
        # Update last login time
        db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        # Convert ObjectId to string for JSON serialization
        user["_id"] = str(user["_id"])
        return user
    
    @classmethod
    def get_user_profile(cls, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by ID."""
        user = db.users.find_one({"_id": user_id})
        if user:
            user["_id"] = str(user["_id"])
            user.pop("password", None)  # Don't return the password hash
        return user

# Create a singleton instance
auth_service = AuthService()
