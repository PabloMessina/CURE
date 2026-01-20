import nltk

REQUIRED_RESOURCES = ['punkt', 'punkt_tab']

def ensure_nltk_resources():
    """Checks for required NLTK resources and downloads them if missing."""
    missing_resources = []
    for resource in REQUIRED_RESOURCES:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            # Optional: print(f"NLTK resource '{resource}' found.")
        except LookupError:
            missing_resources.append(resource)

    if missing_resources:
        print(f"Downloading missing NLTK resources: {missing_resources}...")
        try:
            for resource in missing_resources:
                nltk.download(resource, quiet=True)
            print("NLTK resources downloaded successfully.")
            # Verify after download
            for resource in missing_resources:
                 nltk.data.find(f'tokenizers/{resource}')
            print("NLTK resources verified after download.")
        except Exception as e:
            print(f"Error downloading NLTK data: {e}")
            print(f"  python -m nltk.downloader {' '.join(missing_resources)}")
            # raise RuntimeError("Failed to download required NLTK resources.") from e
    # else:
        # Optional: print("All required NLTK resources found.")

# You could potentially call it once here if you always want it checked on import
# ensure_nltk_resources()
