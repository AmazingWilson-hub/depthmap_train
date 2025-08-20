import os
from pathlib import Path

def check_pairs(split_dir: Path):
    img_dir = split_dir / "images"
    dep_dir = split_dir / "depths"

    if not img_dir.exists() or not dep_dir.exists():
        print(f"[跳過] {split_dir} 缺少 images 或 depths 資料夾")
        return

    img_files = {f.stem for f in img_dir.rglob("*") if f.is_file()}
    dep_files = {f.stem for f in dep_dir.rglob("*") if f.is_file()}

    only_img = sorted(img_files - dep_files)
    only_dep = sorted(dep_files - img_files)
    common   = img_files & dep_files

    print(f"=== {split_dir} ===")
    print(f"images: {len(img_files)}, depths: {len(dep_files)}, matched: {len(common)}")
    if only_img:
        print(f"⚠️  {len(only_img)} 個 images 沒有對應深度，例如: {only_img[:5]}")
    if only_dep:
        print(f"⚠️  {len(only_dep)} 個 depths 沒有對應影像，例如: {only_dep[:5]}")
    if not only_img and not only_dep:
        print("✅ 全部對上")

if __name__ == "__main__":
    root = Path("data_v2")
    for split in ["train", "val"]:
        check_pairs(root / split)
