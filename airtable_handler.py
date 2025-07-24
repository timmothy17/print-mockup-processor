import requests
import logging
import traceback
import os

logger = logging.getLogger(__name__)

# Ensure logging is configured for this module
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Try to import from config, but handle missing config gracefully
try:
    from config import AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME
    logger.info("Successfully imported Airtable configuration")
except ImportError as e:
    logger.warning(f"Failed to import from config.py: {e}")
    # Fallback to environment variables
    AIRTABLE_API_KEY = os.environ.get('AIRTABLE_API_KEY')
    AIRTABLE_BASE_ID = os.environ.get('AIRTABLE_BASE_ID')
    AIRTABLE_TABLE_NAME = os.environ.get('AIRTABLE_TABLE_NAME', 'Ad Mockup Templates')
    logger.info("Using environment variables for Airtable configuration")

class AirtableHandler:
    """Handler for Airtable API operations"""
    
    def __init__(self):
        """Initialize the Airtable handler with API credentials"""
        logger.info("Initializing AirtableHandler")
        
        self.api_key = AIRTABLE_API_KEY
        self.base_id = AIRTABLE_BASE_ID
        self.table_name = AIRTABLE_TABLE_NAME
        
        # Check if essential credentials are available
        if not self.api_key:
            logger.warning("AIRTABLE_API_KEY is not set")
        if not self.base_id:
            logger.warning("AIRTABLE_BASE_ID is not set")
        if not self.table_name:
            logger.warning("AIRTABLE_TABLE_NAME is not set")
            
        # Set up headers for API requests
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        logger.info("AirtableHandler initialized successfully")
    
    def update_record(self, record_id, fields):
        """Update an Airtable record with the provided fields"""
        # Check if we have all required credentials
        if not all([self.api_key, self.base_id, self.table_name]):
            error_msg = "Missing Airtable credentials"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not record_id:
            error_msg = "No record_id provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Correct URL format for Airtable API (no table ID in path)
        url = f"https://api.airtable.com/v0/{self.base_id}/{self.table_name}/{record_id}"
        
        # Log the URL we're using (without sensitive info)
        logger.info(f"Updating Airtable record ID: {record_id}")
        logger.info(f"ID: {record_id}, JSON fields: {fields}")
        data = {
            'fields': fields
        }
        
        try:
            logger.info(f"Sending update request to Airtable for record {record_id}")
            response = requests.patch(url, headers=self.headers, json=data)
            response.raise_for_status()
            logger.info(f"Updated Airtable record {record_id} with {len(fields)} fields")
            return response.json()
        except Exception as e:
            logger.error(f"Error updating Airtable record: {str(e)}")
            logger.error(f"Request data: {data}")
            logger.error(traceback.format_exc())
            raise
    
    def get_record(self, record_id):
        """Get an Airtable record by ID"""
        # Check if we have all required credentials
        if not all([self.api_key, self.base_id, self.table_name]):
            error_msg = "Missing Airtable credentials"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        url = f"https://api.airtable.com/v0/{self.base_id}/{self.table_name}/{record_id}"
        
        try:
            logger.info(f"Retrieving Airtable record {record_id}")
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Retrieved Airtable record {record_id}")
            return response.json()
        except Exception as e:
            logger.error(f"Error retrieving Airtable record: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Module test - this will run when the file is executed directly
if __name__ == "__main__":
    print("Testing AirtableHandler module")
    try:
        handler = AirtableHandler()
        print("AirtableHandler created successfully")
        # Check if credentials are available for proper operation
        if handler.api_key and handler.base_id and handler.table_name:
            print("Airtable credentials are available")
        else:
            print("WARNING: Some Airtable credentials are missing")
    except Exception as e:
        print(f"Error creating AirtableHandler: {e}")
        print(traceback.format_exc())
