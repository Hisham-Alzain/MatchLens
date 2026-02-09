# import requests


# urls = [
#     "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
#     "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
#     "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
# ]
# outputs = [
#     "v10/yolov10n.pt",
#     "v10/yolov10s.pt",
#     "v10/yolov10m.pt",
# ]

# # make sure the folder exists
import os

os.makedirs("v11", exist_ok=True)
# os.makedirs("v10", exist_ok=True)
# os.makedirs("v8", exist_ok=True)
# os.makedirs("v5", exist_ok=True)

# for url, output in zip(urls, outputs):
#     r = requests.get(url, stream=True)
#     r.raise_for_status()

#     with open(output, "wb") as f:
#         for chunk in r.iter_content(chunk_size=8192):
#             if chunk:
#                 f.write(chunk)

#     name = output.split("/")[1]
#     print("Downloaded ", name)


from ultralytics import YOLO
import cv2

print(cv2.__version__)

for name in ("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"):
    pt = f"v11/{name}.pt"
    if not os.path.exists(pt):
        # load and save
        model = YOLO(f"{name}.pt")
        model.save(pt)
        print("Downloaded and saved:", pt)
    else:
        print(pt, "already exists — skipping")


# for name in ("yolov8n", "yolov8s", "yolov8m"):
#     pt = f"v8/{name}.pt"
#     if not os.path.exists(pt):
#         # load and save
#         model = YOLO(f"{name}.pt")
#         model.save(pt)
#         print("Downloaded and saved:", pt)
#     else:
#         print(pt, "already exists — skipping")


# for name in ("yolov5n", "yolov5s", "yolov5m"):
#     pt = f"v5/{name}.pt"
#     if not os.path.exists(pt):
#         # load and save
#         model = YOLO(f"{name}.pt")
#         model.save(pt)
#         print("Downloaded and saved:", pt)
#     else:
#         print(pt, "already exists — skipping")
