#!/usr/bin/env python3
"""
Upload ML model files to Supabase Storage manually
"""
import os
import requests
import json
from pathlib import Path

def upload_to_supabase(file_path: str, storage_path: str):
    """Upload a file to Supabase Storage"""
    SUPABASE_URL = os.environ.get('SUPABASE_URL')
    SUPABASE_SERVICE_ROLE = os.environ.get('SUPABASE_SERVICE_ROLE') 
    SUPABASE_BUCKET = os.environ.get('SUPABASE_BUCKET', 'stockdash')
    
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE:
        print("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE environment variables")
        return False
    
    url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{SUPABASE_BUCKET}/{storage_path}"
    
    with open(file_path, 'rb') as f:
        headers = {
            'Authorization': f'Bearer {SUPABASE_SERVICE_ROLE}',
            'apikey': SUPABASE_SERVICE_ROLE,
            'x-upsert': 'true'
        }
        
        response = requests.post(url, headers=headers, data=f.read())
        
        if response.status_code in [200, 201]:
            print(f"✅ Successfully uploaded {file_path} to {storage_path}")
            return True
        else:
            print(f"❌ Failed to upload {file_path}: {response.status_code} - {response.text}")
            return False

def main():
    # Find the latest ensemble model
    ml_out_dir = Path("ml_out")
    ensemble_models = list(ml_out_dir.glob("model_v3_*_ensemble.json"))

    if ensemble_models:
        # Sort by filename (timestamp) and get the latest
        latest_model = sorted(ensemble_models)[-1]
        model_name = latest_model.name
        storage_path = f"ml/models/{model_name}"

        print(f"Uploading latest ensemble model: {latest_model}")
        if upload_to_supabase(str(latest_model), storage_path):
            # Update latest.json to point to this model
            latest_content = {
                "path": storage_path,
                "version": model_name.replace('.json', '')
            }
            latest_file = ml_out_dir / "latest.json"
            with open(latest_file, 'w') as f:
                json.dump(latest_content, f)

            # Upload the updated latest.json
            upload_to_supabase(str(latest_file), "ml/models/latest.json")
        else:
            print("Failed to upload model")
    else:
        print("No ensemble models found in ml_out/")

        # Fallback to v1 model
        model_file = "ml_out/model_v1_202509201843_yf.json"
        if Path(model_file).exists():
            upload_to_supabase(model_file, "ml/models/model_v1_202509201843_yf.json")
        else:
            print(f"Model file {model_file} not found")

        # Upload the latest pointer
        latest_file = "ml_out/latest.json"
        if Path(latest_file).exists():
            upload_to_supabase(latest_file, "ml/models/latest.json")
        else:
            print(f"Latest file {latest_file} not found")

if __name__ == "__main__":
    main()
