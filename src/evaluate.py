# evaluates the trained model on the test set: dice and head circumference error (mm)
# also writes results/test_results.csv, a real vs predicted scatter plot and a few
# overlay images so we can visually check how well the model did
import argparse
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch

from dataset import IMG_H, IMG_W
from train import get_device

matplotlib.use("Agg")  # draw plots to files, no display needed
import matplotlib.pyplot as plt

# keep only the biggest blob in a mask, dropping small stray predictions
def largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if n <= 1:  # nothing but background
        return mask
    k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # index 0 is the background
    return (labels == k).astype(np.uint8)

# fit an ellipse to the mask and return (perimeter in pixels, the ellipse)
def ellipse_perimeter_px(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:  # fitEllipse needs at least 5 points
        return None, None
    ell = cv2.fitEllipse(cnt)
    a, b = ell[1][0] / 2, ell[1][1] / 2  # opencv gives full axes, we want semi-axes
    # there's no exact formula for an ellipse perimeter, Ramanujan's approximation
    # is very accurate for the shapes we have here
    per = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    return per, ell

# dice overlap between a predicted mask p and the ground-truth mask g
def dice_np(p, g, eps=1e-6):
    return (2 * (p * g).sum() + eps) / (p.sum() + g.sum() + eps)

def main():
    # command-line options
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_plots", type=int, default=12)  # how many overlay images to save
    args = ap.parse_args()

    # load the trained model
    device = get_device()
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device).eval()

    # only the test images, the model never saw them in training or model selection
    df = pd.read_csv(Path(args.processed) / "splits.csv")
    df = df[df["split"] == "test"].reset_index(drop=True)
    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    rows = []
    for i, r in df.iterrows():
        # load the scan and the real mask at full resolution
        img = cv2.imread(str(Path(args.raw) / "training_set" / r["filename"]), cv2.IMREAD_GRAYSCALE)
        gt = (cv2.imread(str(Path(args.processed) / "masks" / r["filename"]), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        H, W = img.shape

        # prepare the image exactly like in training, resize and scale to -1..1
        x = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32) / 255.0
        x = torch.from_numpy((x - 0.5) / 0.5)[None, None].to(device)

        # predict the head probability for each pixel
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        prob = cv2.resize(prob, (W, H))  # back to the original resolution
        pred = largest_component((prob > 0.5).astype(np.uint8))

        # from mask to a clinical number, fit an ellipse, take its perimeter and
        # convert pixels to mm using this image's own pixel size
        per_px, ell = ellipse_perimeter_px(pred)
        hc_pred = per_px * r["pixel_size_mm"] if per_px else np.nan
        rows.append({
            "filename": r["filename"],
            "dice": float(dice_np(pred, gt)),
            "hc_true_mm": r["hc_mm"],
            "hc_pred_mm": hc_pred,
            "abs_err_mm": abs(hc_pred - r["hc_mm"]),
        })

        # save a few overlays, green = our ellipse and red = the annotated outline
        if i < args.n_plots and ell is not None:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.ellipse(vis, ell, (0, 255, 0), 2)
            cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1)
            cv2.imwrite(str(out / "overlays" / r["filename"]), vis)

    # summary numbers
    res = pd.DataFrame(rows)
    res.to_csv(out / "test_results.csv", index=False)
    print(f"Test images: {len(res)}")
    print(f"Mean Dice:               {res['dice'].mean():.4f}")
    print(f"Mean absolute error:     {res['abs_err_mm'].mean():.2f} mm")
    print(f"Median absolute error:   {res['abs_err_mm'].median():.2f} mm")

    # scatter plot of real vs predicted head circumference
    # points on the red dashed line would be perfect predictions
    plt.figure(figsize=(5, 5))
    plt.scatter(res["hc_true_mm"], res["hc_pred_mm"], s=8, alpha=0.6)
    lim = [res["hc_true_mm"].min(), res["hc_true_mm"].max()]
    plt.plot(lim, lim, "r--", lw=1)
    plt.xlabel("Real circumference (mm)")
    plt.ylabel("Predicted circumference (mm)")
    plt.title("Head circumference: real vs predicted (test set)")
    plt.tight_layout()
    plt.savefig(out / "hc_scatter.png", dpi=150)

if __name__ == "__main__":
    main()