import time
import functools
from datetime import datetime, timedelta
from typing import Callable, Any, Optional, Dict
import logging

from config import MAX_DAILY_LLM_CALLS
from db import db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)

class RateLimiter:
    """Rate limiter for API calls with daily limits."""
    
    def __init__(self, max_calls: int = MAX_DAILY_LLM_CALLS, window_seconds: int = 86400):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the time window
            window_seconds: Time window in seconds (default: 86400 = 24 hours)
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
    
    def check_limit(self, user_id: str) -> bool:
        """
        Check if the user has exceeded their rate limit.
        
        Args:
            user_id: The ID of the user to check
            
        Returns:
            bool: True if the user is allowed to make a call, False otherwise
            
        Raises:
            RateLimitExceeded: If the user has exceeded their rate limit
        """
        # Get the user's current usage
        user = db.users.find_one({"_id": user_id})
        
        if not user:
            logger.warning(f"User {user_id} not found")
            return True  # Allow the call if user not found (shouldn't happen in normal flow)
        
        # Check if we need to reset the counter (new day)
        now = datetime.utcnow()
        last_usage_date = user.get("last_llm_usage_date")
        
        if last_usage_date and isinstance(last_usage_date, datetime):
            # If we're in a new day, reset the counter
            if now.date() > last_usage_date.date():
                db.users.update_one(
                    {"_id": user_id},
                    {
                        "$set": {
                            "llm_usage_count": 0,
                            "last_llm_usage_date": now
                        }
                    }
                )
                return True
        
        # Check current usage
        current_usage = user.get("llm_usage_count", 0)
        
        if current_usage >= self.max_calls:
            # Calculate when the limit will reset (next day)
            if last_usage_date:
                reset_time = (last_usage_date + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                retry_after = (reset_time - now).total_seconds()
            else:
                retry_after = None
                
            raise RateLimitExceeded(
                f"Daily limit of {self.max_calls} calls exceeded. Please try again tomorrow.",
                retry_after=retry_after
            )
        
        return True
    
    def increment_usage(self, user_id: str) -> int:
        """
        Increment the user's call count.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            int: The new call count
        """
        now = datetime.utcnow()
        
        result = db.users.update_one(
            {"_id": user_id},
            {
                "$inc": {"llm_usage_count": 1},
                "$set": {"last_llm_usage_date": now}
            },
            upsert=False
        )
        
        if result.modified_count == 0:
            # If this is the first call, initialize the counter
            db.users.update_one(
                {"_id": user_id},
                {
                    "$setOnInsert": {
                        "llm_usage_count": 1,
                        "last_llm_usage_date": now
                    }
                },
                upsert=True
            )
        
        # Get the updated count
        user = db.users.find_one({"_id": user_id}, {"llm_usage_count": 1})
        return user.get("llm_usage_count", 0)
    
    def get_usage(self, user_id: str) -> Dict[str, Any]:
        """
        Get the user's current usage information.
        
        Args:
            user_id: The ID of the user
            
        Returns:
            dict: Dictionary containing usage information
        """
        user = db.users.find_one({"_id": user_id}, {"llm_usage_count": 1, "last_llm_usage_date": 1})
        
        if not user:
            return {
                "current_usage": 0,
                "max_calls": self.max_calls,
                "remaining_calls": self.max_calls,
                "reset_time": None,
                "reset_in_seconds": None
            }
        
        current_usage = user.get("llm_usage_count", 0)
        last_usage_date = user.get("last_llm_usage_date")
        
        # Calculate reset time (next day at midnight UTC)
        if last_usage_date and isinstance(last_usage_date, datetime):
            reset_time = (last_usage_date + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            reset_in_seconds = (reset_time - datetime.utcnow()).total_seconds()
            reset_in_seconds = max(0, reset_in_seconds)
        else:
            reset_time = None
            reset_in_seconds = None
        
        return {
            "current_usage": current_usage,
            "max_calls": self.max_calls,
            "remaining_calls": max(0, self.max_calls - current_usage),
            "reset_time": reset_time,
            "reset_in_seconds": reset_in_seconds
        }

# Create a singleton instance
rate_limiter = RateLimiter(max_calls=MAX_DAILY_LLM_CALLS)

def rate_limited(func: Callable) -> Callable:
    """
    Decorator to enforce rate limiting on a function.
    
    The decorated function must have 'user_id' as one of its parameters.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract user_id from kwargs or args
        user_id = kwargs.get('user_id')
        if not user_id and len(args) > 0:
            # Try to get user_id from args (assuming it's the first argument)
            user_id = args[0]
        
        if not user_id:
            raise ValueError("user_id is required for rate limiting")
        
        # Check rate limit
        rate_limiter.check_limit(user_id)
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            
            # Increment the usage counter
            rate_limiter.increment_usage(user_id)
            
            return result
            
        except Exception as e:
            # Don't increment counter for failed requests
            raise
    
    return wrapper
