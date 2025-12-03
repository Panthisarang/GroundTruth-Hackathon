import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from bson import ObjectId

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Now import the modules
from h002_customer_experience_automation.auth import AuthService
from h002_customer_experience_automation.db import db
from h002_customer_experience_automation.config import SECRET_KEY

# Admin configuration
ADMIN_CONFIG = {
    "email": "admin@example.com",
    "password": "Admin@123",  # In production, use a more secure password
    "first_name": "Admin",
    "last_name": "User",
    "phone": "+1234567890",
    "is_admin": True,
    "preferences": {
        "cuisine": ["italian", "indian", "chinese"],
        "price_range": "medium",  # low, medium, high
        "accessibility": ["wheelchair_accessible", "elevator"],
        "atmosphere": ["quiet", "family_friendly"],
        "amenities": ["wifi", "parking", "outdoor_seating"]
    },
    "location": {
        "type": "Point",
        "coordinates": [-73.9857, 40.7484],  # Default to New York coordinates
        "address": "350 5th Ave, New York, NY 10118",
        "last_updated": datetime.utcnow()
    },
    "notification_preferences": {
        "email": True,
        "push": True,
        "promotions": True,
        "newsletter": True
    },
    "created_at": datetime.utcnow(),
    "last_login": None,
    "login_count": 0,
    "is_active": True
}

# Sample locations data
SAMPLE_LOCATIONS = [
    {
        "name": "Central Park",
        "type": "park",
        "description": "Iconic park with walking paths, lakes, and recreational activities.",
        "location": {
            "type": "Point",
            "coordinates": [-73.9654, 40.7829]
        },
        "address": "59th St to 110th St, New York, NY 10022",
        "rating": 4.8,
        "price_level": 0,  # Free
        "opening_hours": {
            "monday": "06:00-23:00",
            "tuesday": "06:00-23:00",
            "wednesday": "06:00-23:00",
            "thursday": "06:00-23:00",
            "friday": "06:00-23:00",
            "saturday": "06:00-23:00",
            "sunday": "06:00-23:00"
        },
        "amenities": ["parking", "restrooms", "picnic_areas", "playground"],
        "accessibility": ["wheelchair_accessible", "service_animals"]
    },
    {
        "name": "New York Public Library",
        "type": "library",
        "description": "Historic library with extensive collections and reading rooms.",
        "location": {
            "type": "Point",
            "coordinates": [-73.9823, 40.7532]
        },
        "address": "476 5th Ave, New York, NY 10018",
        "rating": 4.7,
        "price_level": 0,  # Free
        "opening_hours": {
            "monday": "10:00-18:00",
            "tuesday": "10:00-20:00",
            "wednesday": "10:00-20:00",
            "thursday": "10:00-18:00",
            "friday": "10:00-18:00",
            "saturday": "10:00-18:00",
            "sunday": "13:00-17:00"
        },
        "amenities": ["wifi", "computers", "printing", "study_rooms"],
        "accessibility": ["wheelchair_accessible", "elevator"]
    },
    {
        "name": "Eataly NYC Downtown",
        "type": "restaurant",
        "cuisine": ["italian"],
        "description": "Italian marketplace with restaurants, food counters & a cooking school.",
        "location": {
            "type": "Point",
            "coordinates": [-74.0099, 40.7053]
        },
        "address": "101 Liberty St 3rd Floor, New York, NY 10007",
        "rating": 4.5,
        "price_level": 3,  # $$$
        "opening_hours": {
            "monday": "07:00-23:00",
            "tuesday": "07:00-23:00",
            "wednesday": "07:00-23:00",
            "thursday": "07:00-23:00",
            "friday": "07:00-00:00",
            "saturday": "08:00-00:00",
            "sunday": "08:00-23:00"
        },
        "amenities": ["wifi", "outdoor_seating", "bar", "takeout"],
        "atmosphere": ["casual", "family_friendly", "trendy"]
    }
]

def create_admin_user():
    """Create an admin user with default preferences and sample data."""
    try:
        # Check if admin already exists
        existing_admin = db.users.find_one({"email": ADMIN_CONFIG["email"], "is_admin": True})
        if existing_admin:
            print(f"Admin user with email {ADMIN_CONFIG['email']} already exists.")
            return
        
        # Create admin user
        admin_data = {
            **ADMIN_CONFIG,
            "password": AuthService.hash_password(ADMIN_CONFIG["password"])
        }
        
        # Insert admin user
        admin_id = db.users.insert_one(admin_data).inserted_id
        
        # Create sample locations if they don't exist
        for location in SAMPLE_LOCATIONS:
            # Check if location already exists
            existing_location = db.locations.find_one({"name": location["name"]})
            if not existing_location:
                location["created_at"] = datetime.utcnow()
                location["updated_at"] = datetime.utcnow()
                db.locations.insert_one(location)
        
        print("=" * 50)
        print("ADMIN USER CREATED SUCCESSFULLY")
        print("=" * 50)
        print(f"Email: {ADMIN_CONFIG['email']}")
        print(f"Password: {ADMIN_CONFIG['password']}")
        print("\nDEFAULT PREFERENCES:")
        print(json.dumps(ADMIN_CONFIG["preferences"], indent=2))
        print("\nSAMPLE LOCATIONS ADDED:")
        for loc in SAMPLE_LOCATIONS:
            print(f"- {loc['name']} ({loc['type']}) at {loc['address']}")
        print("\n" + "!" * 60)
        print("IMPORTANT: Change the default password after first login!")
        print("!" * 60)
        
    except Exception as e:
        print(f"Error creating admin user: {str(e)}")

if __name__ == "__main__":
    print("Creating admin user...")
    create_admin_user()
