import os
import urllib.request

def download_models():
    # Base URL for the model weights
    BASE_URL = "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/"
    
    # Files to download
    files = [
        "tiny_face_detector_model-weights_manifest.json",
        "tiny_face_detector_model-shard1",
        "face_landmark_68_model-weights_manifest.json",
        "face_landmark_68_model-shard1",
        "face_recognition_model-weights_manifest.json",
        "face_recognition_model-shard1",
        "face_recognition_model-shard2",
        "ssd_mobilenetv1_model-weights_manifest.json",
        "ssd_mobilenetv1_model-shard1",
        "ssd_mobilenetv1_model-shard2"
    ]
    
    # Destination directory
    dest_dir = os.path.join("static", "models")
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Downloading face-api.js models to {dest_dir}...")
    
    for filename in files:
        url = BASE_URL + filename
        dest_path = os.path.join(dest_dir, filename)
        
        if os.path.exists(dest_path):
            print(f"Already exists: {filename}")
            continue
            
        print(f"Downloading: {filename}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename} from {url}: {e}")
            # Try alternative repository if master fails
            alt_url = "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights/" + filename
            print(f"Trying alternative source: {alt_url}")
            try:
                urllib.request.urlretrieve(alt_url, dest_path)
                print(f"Successfully downloaded {filename} from alt source")
            except Exception as e_alt:
                print(f"Failed alternative download as well: {e_alt}")

if __name__ == "__main__":
    download_models()
