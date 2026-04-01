# create new buckets
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

# Note: Google recommend using gcloud instead of gsutil
# move files from one bucket to another (in parallel with -m flag)
# gsutil -m -u relax-melodies-android mv gs://project-vlm2/H2/afri-mcqa/lingala/*.json gs://project-vlm2/H2/afri-mcqa/lingala/gpt-5.2
gcloud storage mv "gs://project-vlm2/H2/afri-mcqa/lingala/*.json" "gs://project-vlm2/H2/afri-mcqa/lingala/gpt-5.2/"
