export TUNED_MODEL_S3_URI="s3://air-example-data/air-model-checkpoints/dreambooth-finetuning-dog/"

# AWS CLI configurations to speed up ckpt downloads
awsv2 configure set s3.max_concurrent_requests 32
awsv2 configure set default.s3.preferred_transfer_client crt
awsv2 configure set default.s3.target_bandwidth 100Gb/s
awsv2 configure set default.s3.multipart_chunksize 8MB

# Download the tuned model checkpoint
echo "Downloading model checkpoint to $1..."
awsv2 s3 sync $TUNED_MODEL_S3_URI $1 > /dev/null