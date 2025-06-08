from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Salesforce/blip-image-captioning-base",
    cache_dir="./backend/weights",
    resume_download=True,
    max_workers=1,
)

snapshot_download(
    repo_id="distilbert-base-uncased",
    cache_dir="./backend/weights",
    resume_download=True,
    max_workers=1
)