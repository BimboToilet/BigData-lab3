from pipeline import LakehouseManager
import os

def main():
    storage_options = {
        "aws_endpoint_url": os.getenv("S3_ENDPOINT", "http://minio:9000"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "admin"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "password"),
        "aws_region": "us-east-1",
        "aws_allow_http": "true",
        "aws_s3_allow_unsafe_disable_ssl": "true"
    }

    manager = LakehouseManager(storage_options)

    print("Start Bronze...")
    manager.run_bronze("data/flight_data_2018_2024.csv") 

    print("Start Silver...")
    manager.run_silver()

    print("Optimizing Silver...")
    manager.maintenance()

    print("Start Gold...")
    manager.run_gold()
    
    print("Pipeline complete")

if __name__ == "__main__":
    main()