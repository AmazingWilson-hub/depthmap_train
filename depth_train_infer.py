# -*- coding: utf-8 -*-
# 灰階單通道 Depth Regression：訓練 (train) + 推論輸出 .npy (infer)
# - Dataset：image(灰階) ↔ depth(.npy) 同名配對
# - Loss：L1 + SiLog（可調權重）
# - Metrics：RMSE、AbsRel（以有效像素、>0 且 finite 計）
# - Mixed Precision：CUDA 自動啟用，CPU/MPS 自動關閉或用 bfloat16
# - Checkpoint：best.pt（val RMSE 最佳）、last.pt（最近一次）
import os, math, time, argparse, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# -----------------------
# Utils
# -----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def valid_mask_from(gt: torch.Tensor):
    # 有效像素：非 NaN/Inf 且 > 0
    return torch.isfinite(gt) & (gt > 0)

def rmse(pred, gt, mask):
    if mask.sum() == 0: return torch.tensor(float("nan"), device=pred.device)
    return torch.sqrt(F.mse_loss(pred[mask], gt[mask]))

def abs_rel(pred, gt, mask):
    if mask.sum() == 0: return torch.tensor(float("nan"), device=pred.device)
    return ((pred[mask] - gt[mask]).abs() / gt[mask].clamp_min(1e-6)).mean()

def to_device(batch, device):
    x, y, m = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)

# -----------------------
# Dataset：灰階單通道
# -----------------------
class ImageDepthNPY(Dataset):
    def __init__(self, img_dir, dep_dir, img_size=384, max_depth=80.0, aug=False, allow_exts=None):
        self.img_dir = Path(img_dir); self.dep_dir = Path(dep_dir)
        self.img_size = int(img_size); self.max_depth = float(max_depth)
        self.aug = bool(aug)
        if allow_exts is None:
            allow_exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp")

        # 收集影像，依檔名 stem 配對 .npy
        imgs = []
        for p in self.img_dir.iterdir():
            if p.is_file() and p.suffix.lower() in allow_exts: imgs.append(p)
        self.samples = []
        for p in imgs:
            npy = self.dep_dir / (p.stem + ".npy")
            if npy.exists(): self.samples.append((p, npy))
        if not self.samples:
            raise FileNotFoundError(f"No paired samples under {img_dir} & {dep_dir}")

        # 幾何增強（只做等比裁切/翻轉，避免破壞幾何）
        self.geom_train = T.Compose([
            T.RandomResizedCrop(self.img_size, scale=(0.7, 1.0))
        ]) if self.aug else T.Resize((self.img_size, self.img_size))

        # 轉 tensor + normalize（灰階單通道）
        self.to_tensor_norm = T.Compose([
            T.ToTensor(),                     # [1,H,W] in [0,1]
            T.Normalize(mean=(0.5,), std=(0.5,)),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_p, dep_p = self.samples[idx]
        # 讀灰階（若是 16-bit，PIL 會保留；我們先用 8/16 都能 cover 的路徑）
        pil = Image.open(img_p).convert("L")
        pil = self.geom_train(pil)
        x = self.to_tensor_norm(pil)                  # [1,H,W]

        # 讀深度 .npy → resize 到訓練大小
        d = np.load(dep_p).astype(np.float32)         # [H0,W0]
        d = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)  # [1,1,H0,W0]
        d = F.interpolate(d, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)  # [1,H,W]

        m = valid_mask_from(d)
        y = torch.zeros_like(d)
        y[m] = (d[m] / self.max_depth).clamp(0, 1)    # [0,1] 正規化

        # 簡單左右翻（只在 aug=True 時），與影像一致
        if self.aug and random.random() < 0.5:
            x = torch.flip(x, dims=[2])   # 水平翻轉 (W)
            y = torch.flip(y, dims=[2])
            m = torch.flip(m, dims=[2])
        return x, y, m

# -----------------------
# Model：Tiny UNet（單通道輸入）
# -----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        ch = [64,128,256,512]
        self.d1 = DoubleConv(in_ch, ch[0]); self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(ch[0], ch[1]); self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(ch[1], ch[2]); self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(ch[2], ch[3])
        self.u3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2); self.dc3 = DoubleConv(ch[2]*2, ch[2])
        self.u2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2); self.dc2 = DoubleConv(ch[1]*2, ch[1])
        self.u1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2); self.dc1 = DoubleConv(ch[0]*2, ch[0])
        self.head = nn.Conv2d(ch[0], out_ch, 1)
    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))
        x4 = self.d4(self.p3(x3))
        y = self.u3(x4); y = self.dc3(torch.cat([y, x3], dim=1))
        y = self.u2(y);   y = self.dc2(torch.cat([y, x2], dim=1))
        y = self.u1(y);   y = self.dc1(torch.cat([y, x1], dim=1))
        y = torch.sigmoid(self.head(y))   # [0,1]
        return y

# -----------------------
# Loss：L1 + SiLog
# -----------------------
class DepthLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_silog=0.1):
        super().__init__()
        self.w_l1 = float(w_l1); self.w_silog = float(w_silog)
    def silog(self, pred, gt, mask, eps=1e-6, lam=0.85):
        p = pred[mask].clamp_min(eps)
        g = gt[mask].clamp_min(eps)
        d = torch.log(p) - torch.log(g)
        return torch.mean(d**2) - lam*(torch.mean(d)**2)
    def forward(self, pred, gt, mask):
        loss = 0.0
        if self.w_l1 > 0:
            loss = loss + self.w_l1 * F.l1_loss(pred[mask], gt[mask])
        if self.w_silog > 0 and mask.any():
            loss = loss + self.w_silog * self.silog(pred, gt, mask)
        return loss

# -----------------------
# Evaluate
# -----------------------
@torch.no_grad()
def evaluate(model, loader, device, max_depth):
    model.eval()
    rmse_list, absrel_list = [], []
    for batch in loader:
        x, y, m = to_device(batch, device)
        with torch.autocast(device.type if device.type != "cpu" else "cpu",
                            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                            enabled=(device.type != "cpu")):
            p = model(x)                 # [B,1,H,W] in [0,1]
        p_m = p * max_depth; y_m = y * max_depth
        m2 = m & torch.isfinite(y_m) & (y_m > 0)
        rmse_list.append(rmse(p_m, y_m, m2).item())
        absrel_list.append(abs_rel(p_m, y_m, m2).item())
    return float(np.nanmean(rmse_list)), float(np.nanmean(absrel_list))

# -----------------------
# Train
# -----------------------
def train(args):
    set_seed(42)
    device = torch.device(args.device)

    train_set = ImageDepthNPY(args.train_images, args.train_depths, args.img_size, args.max_depth, aug=True)
    val_set   = ImageDepthNPY(args.val_images,   args.val_depths,   args.img_size, args.max_depth, aug=False)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = UNetSmall(in_ch=1).to(device)
    loss_fn = DepthLoss(args.w_l1, args.w_silog)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # cosine w/ warmup
    def lf(e):
        warm = max(3, int(0.1*args.epochs))
        if e < warm: return (e+1)/warm
        prog = (e-warm)/max(1, args.epochs-warm)
        return 0.5*(1+math.cos(math.pi*prog))
    sch = optim.lr_scheduler.LambdaLR(opt, lf)

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    print(f"Device={device}; train={len(train_set)} val={len(val_set)} img_size={args.img_size} batch={args.batch}")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for x,y,m in train_loader:
            x,y,m = to_device((x,y,m), device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type if device.type != "cpu" else "cpu",
                                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                                enabled=(device.type != "cpu")):
                p = model(x)
                loss = loss_fn(p, y, m)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * x.size(0)

        sch.step()
        tr_loss = running / len(train_loader.dataset)
        val_rmse, val_absrel = evaluate(model, val_loader, device, args.max_depth)
        elapsed = time.time() - t0

        print(f"[{epoch+1:03d}/{args.epochs}] train {tr_loss:.4f} | val RMSE {val_rmse:.3f} AbsRel {val_absrel:.3f} | lr {sch.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # save last
        torch.save({"epoch": epoch, "model": model.state_dict()}, outdir / "last.pt")
        # save best by RMSE
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save({"epoch": epoch, "model": model.state_dict()}, outdir / "best.pt")

    print(f"Done. Best val RMSE={best_rmse:.3f}. ckpt: {outdir/'best.pt'}")

# -----------------------
# Infer：輸出 .npy
# -----------------------
@torch.no_grad()
def infer(args):
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNetSmall(in_ch=1).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    # 收集輸入影像
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp")
    in_dir = Path(args.in_images)
    imgs = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not imgs: raise FileNotFoundError(f"No input images in {in_dir}")

    # 單通道 transform
    tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for p in imgs:
        pil = Image.open(p).convert("L")
        W0,H0 = pil.size
        x = tf(pil).unsqueeze(0).to(device)
        with torch.autocast(device.type if device.type != "cpu" else "cpu",
                            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                            enabled=(device.type != "cpu")):
            y = model(x)     # [1,1,h,w] in [0,1]
        y = (y * args.max_depth).squeeze(0)  # metric depth
        y = F.interpolate(y.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False).squeeze(0)
        depth = y.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [H,W]
        np.save(out_dir / (p.stem + ".npy"), depth)
    print(f"Saved {len(imgs)} npy to {out_dir}")

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train")
    tp.add_argument("--train-images", type=str, required=True)
    tp.add_argument("--train-depths", type=str, required=True)
    tp.add_argument("--val-images",   type=str, required=True)
    tp.add_argument("--val-depths",   type=str, required=True)
    tp.add_argument("--img-size", type=int, default=384)
    tp.add_argument("--max-depth", type=float, default=80.0)
    tp.add_argument("--batch", type=int, default=16)
    tp.add_argument("--epochs", type=int, default=50)
    tp.add_argument("--lr", type=float, default=1e-3)
    tp.add_argument("--wd", type=float, default=1e-4)
    tp.add_argument("--workers", type=int, default=4)
    tp.add_argument("--w-l1", type=float, default=1.0)
    tp.add_argument("--w-silog", type=float, default=0.1)
    tp.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    tp.add_argument("--out", type=str, default="./runs/dep_unet")

    ip = sub.add_parser("infer")
    ip.add_argument("--ckpt", type=str, required=True)
    ip.add_argument("--in-images", type=str, required=True)
    ip.add_argument("--out-dir", type=str, default="./pred_npy")
    ip.add_argument("--img-size", type=int, default=384)
    ip.add_argument("--max-depth", type=float, default=80.0)
    ip.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()
    if args.cmd == "train": train(args)
    elif args.cmd == "infer": infer(args)

if __name__ == "__main__":
    import numpy as np  # 放最後，避免未使用時 IDE 警告
    main()
