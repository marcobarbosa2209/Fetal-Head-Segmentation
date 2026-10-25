# trains a U-Net (ResNet34 encoder) to segment the fetal head in ultrasound images
# the best model (highest Dice on the validation set) is saved to checkpoints/best.pt
import argparse
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HC18Dataset

# pick the fastest hardware available: nvidia gpu, apple gpu (mps) or cpu
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# dice = how much the predicted mask overlaps the real one (1.0 is perfect)
def dice_score(logits, target, eps=1e-6):
    # the network outputs raw scores, sigmoid + threshold turns them into a 0/1 mask
    pred = (torch.sigmoid(logits) > 0.5).float()
    # compute the intersection and union of the predicted and target masks
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    # eps avoids dividing by zero on an empty mask
    return ((2 * inter + eps) / (union + eps)).mean().item()


def main():
    # command-line options
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)        # images per training step
    ap.add_argument("--lr", type=float, default=3e-4)   # learning rate
    ap.add_argument("--workers", type=int, default=2)   # parallel data-loading processes
    ap.add_argument("--out", default="checkpoints/best.pt")
    args = ap.parse_args()

    device = get_device()
    print("device:", device)

    # only the training set gets augmentation, validation is left untouched
    train_ds = HC18Dataset(args.raw, args.processed, "train", train=True)
    val_ds = HC18Dataset(args.raw, args.processed, "val")
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.bs, num_workers=args.workers)

    # model, loss and optimiser
    # U-Net with a ResNet34 encoder that starts from ImageNet weights, which helps
    # a lot when there are only a few hundred images
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1).to(device)
    # two losses added together, dice looks at overlap and BCE looks at every pixel
    # together they train more steadily than either one alone
    dice_loss = smp.losses.DiceLoss("binary")
    bce = smp.losses.SoftBCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # slowly lowers the learning rate so the model settles down at the end
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # training loop
    best = 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        # training pass, the model sees every training image once
        model.train()
        tot = 0.0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)                             # the model's guess
            loss = dice_loss(logits, y) + bce(logits, y)  # how wrong it was
            opt.zero_grad()
            loss.backward()                               # work out how to improve
            opt.step()                                    # take a step towards that
            tot += loss.item() * x.size(0)
        sched.step()

        # validation pass, no learning here, just measuring on unseen images
        model.eval()
        scores = []
        with torch.no_grad():
            for x, y in val_dl:
                scores.append(dice_score(model(x.to(device)), y.to(device)))
        val_dice = sum(scores) / len(scores)
        print(f"epoch {epoch:02d} | train loss {tot/len(train_ds):.4f} | val dice {val_dice:.4f}")

        # keep a copy of the model whenever it beats its previous best
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), args.out)
            print(f"  -> new best model (dice {best:.4f})")

if __name__ == "__main__":
    main()