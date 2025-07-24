import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# S3 Configuration
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'drool-ad-mockups')
AWS_REGION = os.environ.get('AWS_REGION', 'eu-north-1')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

# Airtable Configuration
AIRTABLE_API_KEY = os.environ.get('AIRTABLE_API_KEY')
AIRTABLE_BASE_ID = os.environ.get('AIRTABLE_BASE_ID')
AIRTABLE_TABLE_NAME = os.environ.get('AIRTABLE_TABLE_NAME', 'Ad Mockups')

# Renderform Configuration
# Renderform Configuration
RENDERFORM_API_KEY = os.environ.get('RENDERFORM_API_KEY')
RENDERFORM_TEMPLATE_ID = os.environ.get('RENDERFORM_TEMPLATE_ID')
RENDERFORM_TEMPLATE_ID_1_1 = os.environ.get('RENDERFORM_TEMPLATE_ID_1_1')
RENDERFORM_TEMPLATE_ID_4_5 = os.environ.get('RENDERFORM_TEMPLATE_ID_4_5')
RENDERFORM_TEMPLATE_ID_9_16 = os.environ.get('RENDERFORM_TEMPLATE_ID_9_16')
# Flask Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key')
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
PORT = int(os.environ.get('PORT', 5000))
