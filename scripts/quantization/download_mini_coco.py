import random
import requests
from pathlib import Path

DATA_DIR = Path("./calibration_data")
NUM_IMAGES = 100
COCO_VAL_URL = "http://images.cocodataset.org/val2017/"
MAX_COCO_ID = 600_000   # safe upper bound
TIMEOUT = 10

random.seed(42)
DATA_DIR.mkdir(parents=True, exist_ok=True)

downloaded = 0
attempts = 0

while downloaded < NUM_IMAGES:
    img_id = random.randint(1, MAX_COCO_ID)
    img_name = f"{img_id:012d}.jpg"
    img_path = DATA_DIR / img_name

    if img_path.exists():
        continue

    img_url = f"{COCO_VAL_URL}{img_name}"
    attempts += 1

    try:
        r = requests.get(img_url, timeout=TIMEOUT)
        if r.status_code == 200:
            img_path.write_bytes(r.content)
            downloaded += 1
            print(f"[{downloaded}/{NUM_IMAGES}] Downloaded {img_name}")
    except Exception:
        pass  # ignore 404 / network noise

print(f"\nDone. Downloaded {downloaded} images in {attempts} attempts.")
