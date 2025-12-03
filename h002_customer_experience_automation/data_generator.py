import os
import json
import random
from datetime import datetime, timedelta
from faker import Faker
from pymongo import MongoClient
from typing import List, Dict, Any
import logging
from pathlib import Path

from h002_customer_experience_automation.config import (
    MONGODB_URI, DB_NAME, USERS_COLLECTION, ORDERS_COLLECTION,
    STORES_COLLECTION, COUPONS_COLLECTION, POLICIES_DIR
)
from h002_customer_experience_automation.db import db
from h002_customer_experience_automation.pii_masker import mask_pii

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker
fake = Faker()

class DataGenerator:
    """Generate synthetic data for testing and development."""
    
    def __init__(self):
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[DB_NAME]
        
        # Store categories and products
        self.categories = [
            "Coffee & Tea", "Bakery", "Breakfast", "Lunch", "Desserts",
            "Snacks", "Cold Drinks", "Hot Drinks", "Sandwiches", "Salads"
        ]
        
        self.products = {
            "Coffee & Tea": [
                "Espresso", "Cappuccino", "Latte", "Americano", "Mocha",
                "Flat White", "Macchiato", "Cold Brew", "Iced Coffee", "Chai Latte"
            ],
            "Bakery": [
                "Croissant", "Muffin", "Cinnamon Roll", "Scone", "Bagel",
                "Danish Pastry", "Brownie", "Cookie", "Cake Slice"
            ],
            # Add more product categories as needed
        }
        
        # Store types and locations
        self.store_types = ["Cafe", "Bakery", "Coffee Shop", "Dessert Bar"]
        self.locations = [
            (40.7128, -74.0060, "New York, NY"),  # New York
            (34.0522, -118.2437, "Los Angeles, CA"),  # Los Angeles
            (41.8781, -87.6298, "Chicago, IL"),  # Chicago
            (29.7604, -95.3698, "Houston, TX"),  # Houston
            (33.4484, -112.0740, "Phoenix, AZ")  # Phoenix
        ]
    
    def generate_user(self, mask_pii_data: bool = True) -> Dict[str, Any]:
        """Generate a synthetic user profile."""
        first_name = fake.first_name()
        last_name = fake.last_name()
        email = f"{first_name.lower()}.{last_name.lower()}@example.com"
        
        user = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": fake.phone_number(),
            "address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "zip_code": fake.zipcode(),
                "country": "USA"
            },
            "preferences": {
                "favorite_categories": random.sample(
                    self.categories, 
                    k=random.randint(1, 3)
                ),
                "dietary_restrictions": random.choices(
                    ["None", "Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Nut-Free"],
                    k=random.randint(0, 2)
                ),
                "preferred_store_type": random.choice(self.store_types)
            },
            "created_at": datetime.utcnow(),
            "last_login": None,
            "llm_usage_count": 0,
            "is_active": True
        }
        
        # Generate a password hash (in a real app, use proper password hashing)
        user["password"] = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"  # "password"
        
        # Mask PII if requested
        if mask_pii_data:
            masked_user = user.copy()
            masked_user["email"], _ = mask_pii(user["email"])
            masked_user["phone"], _ = mask_pii(user["phone"])
            masked_user["address"]["street"], _ = mask_pii(user["address"]["street"])
            return masked_user
        
        return user
    
    def generate_store(self, location: tuple) -> Dict[str, Any]:
        """Generate a synthetic store."""
        lat, lon, city = location
        store_type = random.choice(self.store_types)
        store_name = f"{fake.company()} {store_type}"
        
        return {
            "name": store_name,
            "type": store_type,
            "location": {
                "type": "Point",
                "coordinates": [lon, lat]  # GeoJSON uses [longitude, latitude]
            },
            "address": {
                "street": fake.street_address(),
                "city": city,
                "state": city.split(", ")[1] if ", " in city else "CA",
                "zip_code": fake.zipcode(),
                "country": "USA"
            },
            "contact": {
                "phone": fake.phone_number(),
                "email": f"contact@{store_name.lower().replace(' ', '')}.com"
            },
            "operating_hours": {
                "monday": "7:00 AM - 9:00 PM",
                "tuesday": "7:00 AM - 9:00 PM",
                "wednesday": "7:00 AM - 9:00 PM",
                "thursday": "7:00 AM - 10:00 PM",
                "friday": "7:00 AM - 10:00 PM",
                "saturday": "8:00 AM - 10:00 PM",
                "sunday": "8:00 AM - 8:00 PM"
            },
            "amenities": random.sample(
                ["WiFi", "Outdoor Seating", "Pet Friendly", "Wheelchair Accessible", "Drive Thru"],
                k=random.randint(1, 4)
            ),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "is_active": True
        }
    
    def generate_coupon(self, store_id: str) -> Dict[str, Any]:
        """Generate a synthetic coupon for a store."""
        discount_types = ["PERCENT_OFF", "AMOUNT_OFF", "BUY_ONE_GET_ONE"]
        discount_type = random.choice(discount_types)
        
        if discount_type == "PERCENT_OFF":
            discount_value = random.choice([10, 15, 20, 25, 30, 50])
            description = f"{discount_value}% off your purchase"
        elif discount_type == "AMOUNT_OFF":
            discount_value = round(random.uniform(1, 10), 2)
            description = f"${discount_value} off your purchase"
        else:  # BUY_ONE_GET_ONE
            discount_value = 0
            description = "Buy one get one free"
        
        # Random validity period (1-30 days from now)
        valid_from = datetime.utcnow()
        valid_until = valid_from + timedelta(days=random.randint(1, 30))
        
        return {
            "store_id": store_id,
            "code": fake.bothify(text="????-####").upper(),
            "description": description,
            "discount_type": discount_type,
            "discount_value": discount_value,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "min_purchase_amount": round(random.uniform(5, 20), 2),
            "max_discount": round(random.uniform(5, 15), 2) if discount_type == "PERCENT_OFF" else None,
            "is_active": True,
            "usage_limit": random.choice([10, 25, 50, 100, None]),
            "times_used": 0
        }
    
    def generate_order(self, user_id: str, store_id: str) -> Dict[str, Any]:
        """Generate a synthetic order."""
        # Random order date (within last 90 days)
        order_date = datetime.utcnow() - timedelta(days=random.randint(0, 90))
        
        # Generate order items
        num_items = random.randint(1, 5)
        items = []
        total_amount = 0.0
        
        for _ in range(num_items):
            category = random.choice(list(self.products.keys()))
            product = random.choice(self.products[category])
            quantity = random.randint(1, 3)
            price = round(random.uniform(2.5, 8.99), 2)
            
            item = {
                "product": product,
                "category": category,
                "quantity": quantity,
                "unit_price": price,
                "total_price": round(price * quantity, 2)
            }
            
            items.append(item)
            total_amount += item["total_price"]
        
        # Apply random tax and service charge
        tax = round(total_amount * 0.08, 2)  # 8% tax
        service_charge = round(total_amount * 0.1, 2)  # 10% service charge
        
        return {
            "user_id": user_id,
            "store_id": store_id,
            "items": items,
            "subtotal": round(total_amount, 2),
            "tax": tax,
            "service_charge": service_charge,
            "total_amount": round(total_amount + tax + service_charge, 2),
            "order_date": order_date,
            "status": random.choice(["completed", "completed", "completed", "cancelled"]),
            "payment_method": random.choice(["credit_card", "debit_card", "mobile_wallet"]),
            "delivery_address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "zip_code": fake.zipcode(),
                "country": "USA"
            },
            "notes": random.choice(["", "Please include extra napkins", "No onions please", "Allergic to peanuts"])
        }
    
    def create_sample_policies(self) -> None:
        """Create sample policy documents."""
        policies = [
            {
                "title": "Return Policy",
                "content": """
                RETURN POLICY
                
                We want you to be completely satisfied with your purchase. If you're not happy with your order, 
                you may return it within 30 days of purchase for a full refund or exchange.
                
                Conditions:
                - Item must be in its original condition
                - Original receipt or proof of purchase is required
                - Perishable items cannot be returned
                - Shipping fees are non-refundable
                """
            },
            {
                "title": "Privacy Policy",
                "content": """
                PRIVACY POLICY
                
                We are committed to protecting your privacy. This policy explains how we collect, use, and safeguard 
                your personal information when you use our services.
                
                Information We Collect:
                - Contact information (name, email, phone number)
                - Payment details (processed securely by our payment processor)
                - Order history and preferences
                - Device and usage information
                
                How We Use Your Information:
                - Process and fulfill your orders
                - Improve our products and services
                - Send promotional offers (with your consent)
                - Comply with legal obligations
                """
            },
            # Add more policies as needed
        ]
        
        # Create policies directory if it doesn't exist
        os.makedirs(POLICIES_DIR, exist_ok=True)
        
        # Save policies to files
        for policy in policies:
            filename = f"{policy['title'].lower().replace(' ', '_')}.txt"
            filepath = os.path.join(POLICIES_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(policy['content'])
            
            logger.info(f"Created policy file: {filepath}")
    
    def generate_sample_data(self, num_users: int = 10, num_locations: int = 5, orders_per_user: int = 3) -> None:
        """Generate sample data for testing."""
        logger.info("Clearing existing data...")
        self.db[USERS_COLLECTION].delete_many({})
        self.db[STORES_COLLECTION].delete_many({})
        self.db[COUPONS_COLLECTION].delete_many({})
        self.db[ORDERS_COLLECTION].delete_many({})
        self.db[CHAT_HISTORY_COLLECTION].delete_many({})
        self.db[EMBEDDINGS_COLLECTION].delete_many({})

        logger.info(f"Generating {num_locations} locations...")
        locations = []
        for _ in range(num_locations):
            location = self.generate_store((random.uniform(37.0, 38.0), random.uniform(-122.0, -121.0)))
            result = self.db[STORES_COLLECTION].insert_one(location)
            location['_id'] = result.inserted_id
            locations.append(location)

            num_coupons = random.randint(1, 3)
            for _ in range(num_coupons):
                coupon = self.generate_coupon(str(result.inserted_id))
                self.db[COUPONS_COLLECTION].insert_one(coupon)
        
        logger.info(f"Generating {num_users} users with {orders_per_user} orders each...")
        for i in range(num_users):
            user = self.generate_user()
            result = self.db[USERS_COLLECTION].insert_one(user)
            user_id = result.inserted_id
            
            for j in range(orders_per_user):
                location = random.choice(locations)
                order = self.generate_order(user_id, location['_id'])
                self.db[ORDERS_COLLECTION].insert_one(order)
            
            if (i + 1) % 10 == 0 or i == num_users - 1:
                logger.info(f"Generated {i + 1}/{num_users} users")
        
        logger.info("Creating sample policy documents...")
        self.create_sample_policies()
        
        logger.info("Creating database indexes...")
        try:
            self.db[USERS_COLLECTION].create_index("email", unique=True)
            self.db[STORES_COLLECTION].create_index([("location", "2dsphere")])
            self.db[STORES_COLLECTION].create_index("type")  # For filtering by location type
            self.db[COUPONS_COLLECTION].create_index("store_id")
            self.db[ORDERS_COLLECTION].create_index("user_id")
            self.db[CHAT_HISTORY_COLLECTION].create_index("user_id")
            self.db[EMBEDDINGS_COLLECTION].create_index("content_type")
            self.db[EMBEDDINGS_COLLECTION].create_index("content_id")
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
        
        logger.info("✅ Data generation completed successfully!")
        
        logger.info("\nSample data generation complete!")
        logger.info(f"Users: {self.db[USERS_COLLECTION].count_documents({})}")
        logger.info(f"Locations: {self.db[STORES_COLLECTION].count_documents({})}")
        logger.info(f"Coupons: {self.db[COUPONS_COLLECTION].count_documents({})}")
        logger.info(f"Orders: {self.db[ORDERS_COLLECTION].count_documents({})}")
        logger.info(f"Chat History: {self.db[CHAT_HISTORY_COLLECTION].count_documents({})}")
        logger.info(f"Embeddings: {self.db[EMBEDDINGS_COLLECTION].count_documents({})}")
        print("============================")

if __name__ == "__main__":
    generator = DataGenerator()
    generator.generate_sample_data(
        num_users=20,       # Generate 20 users
        num_locations=10,   # Generate 10 locations
        orders_per_user=5   # 1-5 orders per user
    )
