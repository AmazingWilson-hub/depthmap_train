# -*- coding: utf-8 -*-
# 將 EVIL_*/*imag*/{*.png,jpg} 與 highway_sunny_day_*/depth_output/{*.npy} 以「相同時間戳」做配對，
# 再把同名(檔名stem)的影像與深度成對拷成 data/{train,val}/{images,depths}/
# 目的地檔名加上 session 前綴，避免 000000.* 撞名。含 tqdm 進度條。
#python pair_by_timestamp_split.py --root .\highway_sunny_day --img-subdir image --val-last 5 --out .\data_v2
import argparse, re, datetime, shutil
from pathlib import Path
from tqdm import tqdm

TS_RE = re.compile(r".*_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$")  # 抓資料夾最後的時間戳

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def parse_args():
    ap = argparse.ArgumentParser(description="Pair EVIL images with highway depth by timestamp, then split train/val.")
    ap.add_argument("--root", required=True, help="指到 DEPTHMAP_TRAIN 的上層，如 D:/code/depthmap_train/DEPTHMAP_TRAIN")
    ap.add_argument("--img-subdir", default="image", help="EVIL_* 內放原圖的子資料夾名稱 (預設 imag)")
    ap.add_argument("--val-last", type=int, default=2, help="最後 N 個 session 當驗證（預設 2）")
    ap.add_argument("--out", default="data", help="輸出根資料夾（預設 data/）")
    ap.add_argument("--dry-run", action="store_true", help="只預覽不動作")
    ap.add_argument("--copy", action="store_true", help="強制 copy，建議在 Windows 使用（預設就會 copy）")
    return ap.parse_args()

def parse_ts(name: str):
    m = TS_RE.match(name)
    if not m: return None
    return datetime.datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S")

def ensure_dirs(base: Path):
    for sp in ("train", "val"):
        (base/sp/"images").mkdir(parents=True, exist_ok=True)
        (base/sp/"depths").mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, do_copy: bool, dry: bool):
    if dry:
        print(("COPY " if do_copy else "LINK ") + f"{src} -> {dst}")
        return
    if dst.exists():  # 已存在就跳過
        return
    # Windows 沒 symlink 權限常見，預設直接 copy 最穩
    shutil.copy2(src, dst)

def main():
    args = parse_args()
    ROOT = Path(args.root)

    # 1) 掃 EVIL_*（影像）與 highway_sunny_day_*（深度）
    evil = {}
    for p in ROOT.glob("EVIL_*"):
        if not p.is_dir(): continue
        ts = parse_ts(p.name)
        if not ts: continue
        img_dir = p / args.img_subdir
        if img_dir.is_dir():
            evil[ts] = img_dir

    depth = {}
    for p in ROOT.glob("highway_sunny_day_*"):
        if not p.is_dir(): continue
        ts = parse_ts(p.name)
        if not ts: continue
        ddir = p / "depth_output"
        if ddir.is_dir():
            depth[ts] = ddir

    if not evil or not depth:
        print(f"找不到 EVIL_* 或 highway_sunny_day_* 結構，請確認 --root 路徑。")
        return

    # 2) 取得「兩邊同時存在」的時間戳，排序（舊→新）
    common_ts = sorted(set(evil.keys()) & set(depth.keys()))
    if len(common_ts) == 0:
        print("沒有找到同時具有影像與深度的時間戳 (EVIL_* 與 highway_* 沒交集)。")
        return

    # 切 train/val（最後 N 個 session 當 val）
    k = max(1, min(args.val_last, len(common_ts)-1))
    train_ts, val_ts = common_ts[:-k], common_ts[-k:]

    print(f"共有 session(交集) = {len(common_ts)} → train={len(train_ts)} / val={len(val_ts)}")
    if args.dry_run:
        print("Train（範例3個）→", [t.strftime("%Y-%m-%d-%H-%M-%S") for t in train_ts[:3]], "...")
        print("Val（最後） →", [t.strftime("%Y-%m-%d-%H-%M-%S") for t in val_ts])

    # 3) 建立輸出資料夾
    OUT = Path(args.out)
    ensure_dirs(OUT)

    def process(ts_list, split):
        img_out = OUT/split/"images"
        dep_out = OUT/split/"depths"
        total = 0
        for ts in tqdm(ts_list, desc=f"{split}: sessions", unit="sess"):
            img_dir = evil[ts]
            dep_dir = depth[ts]
            sname_img = img_dir.parent.name         # EVIL_YYYY-...
            sname_dep = dep_dir.parent.name         # highway_sunny_day_YYYY-...

            # 以「深度檔」為基準，找同名影像
            npys = sorted(dep_dir.glob("*.npy"))
            if not npys: continue

            # 建立影像索引（該 session）
            img_index = {}
            for p in img_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    img_index.setdefault(p.stem, p)

            for npy in tqdm(npys, leave=False, desc=f"{sname_dep}", unit="file"):
                stem = npy.stem
                img = img_index.get(stem, None)
                if img is None:
                    # 找不到對應原始影像就跳過
                    continue
                # 目的地檔名：用「深度 session 名」當前綴，兩邊一致，避免撞名也能一一對上
                dst_img = img_out / f"{sname_dep}__{img.name}"
                dst_npy = dep_out / f"{sname_dep}__{npy.name}"
                link_or_copy(img, dst_img, do_copy=True, dry=args.dry_run)   # Windows → copy 最穩
                link_or_copy(npy, dst_npy, do_copy=True, dry=args.dry_run)
                total += 1
        return total

    n_tr = process(train_ts, "train")
    n_va = process(val_ts, "val")
    print(("Dry-run 預覽完成。" if args.dry_run else "完成。") + f" 成功配對 train={n_tr}、val={n_va} 張。")

if __name__ == "__main__":
    main()
