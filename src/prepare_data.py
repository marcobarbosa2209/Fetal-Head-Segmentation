# turns the raw HC18 data into something a segmentation model can learn from
# the annotations are only ellipse outlines, so we fill them in to get real masks
# we also split the images into train / val / test, grouped by patient
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# read an annotation (just an ellipse outline) and return a filled 0/1 mask
def fill_annotation(path: Path) -> np.ndarray:
    # load as grayscale and binarise, anything bright is part of the outline
    ann = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    ann = (ann > 127).astype(np.uint8)

    # find the outline as a contour and paint everything inside it
    contours, _ = cv2.findContours(ann, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(ann)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
    return mask

def main():
    # command-line options
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)  # fixed seed, same split every run
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    # load the table with pixel sizes and the real head circumference
    df = pd.read_csv(raw / "training_set_pixel_size_and_HC.csv")
    df.columns = ["filename", "pixel_size_mm", "hc_mm"]  # shorter names
    # filenames look like "001_HC.png" or "001_2HC.png", the part before the
    # first underscore is the patient (one patient can have several scans)
    df["patient"] = df["filename"].str.split("_").str[0]

    # build one filled mask per image
    kept = []
    for fn in df["filename"]:
        ann_path = raw / "training_set" / fn.replace(".png", "_Annotation.png")
        if not ann_path.exists():
            print(f"[warning] no annotation for: {fn}")
            kept.append(False)
            continue
        mask = fill_annotation(ann_path)
        cv2.imwrite(str(out / "masks" / fn), mask * 255)  # saved as 0/255 so it's viewable
        kept.append(True)
    df = df[kept].reset_index(drop=True)

    # split by patient, not by image
    # if scans of the same patient ended up in train and test, the model would be
    # tested on anatomy it already saw and the score would look better than it is
    # first cut: 70% train, 30% for the rest
    gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=args.seed)
    train_idx, rest_idx = next(gss.split(df, groups=df["patient"]))
    # second cut: split the rest in half, 15% validation and 15% test
    rest = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=args.seed)
    val_rel, test_rel = next(gss2.split(rest, groups=rest["patient"]))

    df["split"] = ""
    df.loc[df.index[train_idx], "split"] = "train"
    df.loc[rest.index[val_rel], "split"] = "val"
    df.loc[rest.index[test_rel], "split"] = "test"

    # save the split and print a quick summary
    df.to_csv(out / "splits.csv", index=False)
    print(df["split"].value_counts())
    print("Patients per split:", df.groupby("split")["patient"].nunique().to_dict())

if __name__ == "__main__":
    main()
