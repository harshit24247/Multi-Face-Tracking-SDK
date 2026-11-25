import argparse, copy, math, os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from common import LandmarkNet, wing_loss, nme_per_point, REGIONS
from dataset import FaceLM


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval(); errs = []
    for x, y in dl:
        errs.append(nme_per_point(model(x.to(dev)).float(), y.to(dev)).cpu())
    e = torch.cat(errs)
    nme = e.mean(1)
    return dict(nme=100 * nme.mean().item(), fail=100 * (nme > .08).float().mean().item(),
                **{k: 100 * e[:, a:b].mean().item() for k, (a, b) in REGIONS.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/annotations.json")
    ap.add_argument("--backbone", default="mobilenetv2_050")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/exp1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tl = DataLoader(FaceLM(a.ann, "train", True), a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(FaceLM(a.ann, "val", False), 128, num_workers=a.workers)
    model = LandmarkNet(a.backbone).to(dev)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total, warm = a.epochs * len(tl), 3 * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else .5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    tb, best, step = SummaryWriter(a.out), 1e9, 0
    for ep in range(a.epochs):
        model.train(); run = 0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                out = model(x)
            loss = wing_loss(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            d = min(.999, (1 + step) / (10 + step)); step += 1
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            run += loss.item()
        m = evaluate(ema, vl, dev)
        tb.add_scalar("train/loss", run / len(tl), ep)
        for k, v in m.items(): tb.add_scalar(f"val/{k}", v, ep)
        print(f"ep {ep:3d} loss {run/len(tl):.4f} | " + " ".join(f"{k} {v:.2f}" for k, v in m.items()), flush=True)
        if m["nme"] < best:
            best = m["nme"]
            torch.save(dict(ema=ema.state_dict(), backbone=a.backbone, metrics=m), f"{a.out}/best.pt")
    print("best val NME %:", best)


if __name__ == "__main__":
    main()