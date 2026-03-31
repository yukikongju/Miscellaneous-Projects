gcloud storage buckets create gs://project-vlm2 \
    --default-storage-class=STANDARD \
    --location=northamerica-northeast1 \
    --uniform-bucket-level-access \
    --public-access-prevention \
    --project=relax-melodies-android


# Check which account is active
gcloud auth list

# Test a write
echo "test" | gsutil cp - gs://project-vlm2/H2/afri-mcqa/linguala/test.txt
