# s3_helpers.py
import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

def _get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET,
        region_name=AWS_REGION,
    )

def upload_bytes_to_s3(data: bytes, bucket: str, key: str):
    s3 = _get_s3_client()
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=data, ACL="public-read", ContentType="image/png")
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"S3 upload failed: {e}")

def get_s3_public_url(bucket: str, key: str) -> str:
    region = os.getenv("AWS_REGION", "us-east-1")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
