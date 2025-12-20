import requests
import zipfile
import os
import glob

# Direct download link
url = "https://drive.google.com/uc?id=1yGpR_vneP3f3KC4xynlIT_q7WaniC-Nh"

# Folder where dataset will be stored
dataset_folder = "../dataset"
os.makedirs(dataset_folder, exist_ok=True)

# Determine ZIP file name dynamically
existing_zips = glob.glob("archive*.zip")
if existing_zips:
    # Use the first existing zip (handles "archive (1).zip")
    zip_path = existing_zips[0]
    print(f"Found existing ZIP: {zip_path}")
else:
    zip_path = "archive.zip"
    print("Downloading dataset...")
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

# Extract ZIP
print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_folder)

print("Dataset ready in", dataset_folder)
