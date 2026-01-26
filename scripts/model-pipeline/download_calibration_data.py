import argparse
import asyncio
from pathlib import Path

import aiohttp
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "calibration_data"
IDS_FILE = SCRIPT_DIR / "coco_image_ids.txt"
COCO_VAL_URL = "http://images.cocodataset.org/val2017/"
TIMEOUT = aiohttp.ClientTimeout(total=30)
MAX_CONCURRENT = 10


async def download_image(
    session: aiohttp.ClientSession,
    img_id: int,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    stats: dict,
) -> bool:
    img_name = f"{img_id:012d}.jpg"
    img_path = output_dir / img_name

    if img_path.exists():
        stats["skipped"] += 1
        pbar.update(1)
        return True

    async with semaphore:
        url = f"{COCO_VAL_URL}{img_name}"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    img_path.write_bytes(content)
                    stats["downloaded"] += 1
                    pbar.update(1)
                    return True
                else:
                    stats["failed"] += 1
                    pbar.update(1)
                    return False
        except Exception:
            stats["failed"] += 1
            pbar.update(1)
            return False


async def download_calibration_images(output_dir: Path, count: int) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ids = [int(line.strip()) for line in IDS_FILE.read_text().splitlines() if line.strip()]
    image_ids = all_ids[:count]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    with tqdm(total=len(image_ids), desc="Downloading", unit="img") as pbar:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            tasks = [
                download_image(session, img_id, output_dir, semaphore, pbar, stats)
                for img_id in image_ids
            ]
            await asyncio.gather(*tasks)

    print(f"Done. Downloaded: {stats['downloaded']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    return stats["downloaded"] + stats["skipped"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO images for INT8 calibration")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of images to download (default: 100)",
    )
    args = parser.parse_args()

    asyncio.run(download_calibration_images(args.output_dir, args.count))


if __name__ == "__main__":
    main()
