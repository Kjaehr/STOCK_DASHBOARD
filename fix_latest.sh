#!/bin/bash
# Fix latest.json in Supabase Storage to point to the correct model

# Check if environment variables are set
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_SERVICE_ROLE" ]; then
    echo "Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE environment variables"
    exit 1
fi

SUPABASE_BUCKET="stockdash"
LATEST_FILE="ml_out/latest.json"

echo "Uploading latest.json to Supabase Storage..."

curl -sS -X POST \
  "${SUPABASE_URL%/}/storage/v1/object/${SUPABASE_BUCKET}/ml/models/latest.json" \
  -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE}" \
  -H "apikey: ${SUPABASE_SERVICE_ROLE}" \
  -H "x-upsert: true" \
  --data-binary @"${LATEST_FILE}" \
  -o /dev/null -w '%{http_code}' | tee /tmp/upcode

code=$(cat /tmp/upcode)
if [ "$code" = "200" ] || [ "$code" = "201" ]; then
    echo "✅ Successfully uploaded latest.json"
    
    # Verify the upload
    echo "Verifying upload..."
    curl -s "${SUPABASE_URL%/}/storage/v1/object/${SUPABASE_BUCKET}/ml/models/latest.json"
else
    echo "❌ Upload failed with code: $code"
    exit 1
fi
