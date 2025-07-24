import boto3
import logging
import requests
from io import BytesIO
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME

logger = logging.getLogger(__name__)

class S3Handler:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        self.bucket_name = S3_BUCKET_NAME

    def upload_file_obj(self, file_obj, key, content_type=None):
        """Upload a file-like object to S3"""
        try:
            extra_args = {'ACL': 'public-read'}
            if content_type:
                extra_args['ContentType'] = content_type

            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                key,
                ExtraArgs=extra_args
            )
            
            # Generate S3 URL
            url = f"https://{self.bucket_name}.s3.{AWS_REGION}.amazonaws.com/{key}"
            logger.info(f"Uploaded to S3: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise

    def download_file(self, url):
        """Download a file from a URL and return as BytesIO"""
        try:
            logger.info(f"Downloading file from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully downloaded {len(response.content)} bytes")
            return BytesIO(response.content)
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise
