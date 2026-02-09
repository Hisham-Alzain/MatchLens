import numpy as np
import json


def save_results_txt(
    boxes, scores, classes, names, filename="results.txt", append=False
):
    """
    Save YOLO results to TXT.
    Each line: class_name conf x1 y1 x2 y2
    append=True will add to the file, otherwise overwrite.
    """
    mode = "a" if append else "w"
    with open(filename, mode) as f:
        for box, conf, cls in zip(boxes, scores, classes):
            class_name = names[cls]
            x1, y1, x2, y2 = box
            f.write(f"{class_name} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")


def save_results_json(
    boxes, scores, classes, names, filename="results.json", append=False
):
    """
    Save YOLO results to JSON.
    append=True will load existing JSON and append new results.
    """
    if append:
        try:
            with open(filename, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []
    else:
        data = []

    for box, conf, cls in zip(boxes, scores, classes):
        data.append(
            {"class": names[cls], "conf": float(conf), "bbox": [float(x) for x in box]}
        )

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
