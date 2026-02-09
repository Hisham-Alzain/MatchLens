from ultralytics import YOLO
import torch
import os

if __name__ == "__main__":
    # Check GPU
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Check CPU
    print(os.cpu_count())
    # Optimal workers
    WORKERS = min(os.cpu_count() // 2, 4)

    # Choose YOLOv model
    # model = YOLO("yolo11s.pt")
    # v10n (nano) for speed,
    # v10s (small) for balance,
    # v10m (medium) for accuracy

    # Train the model
    # model.train(
    #     data="dataset/data.yaml",  # path to data.yaml
    #     epochs=120,  # number of iterations
    #     imgsz=960,  # input image size
    #     batch=12,  # based on GPU memory
    #     device=0,  # force use GPU (Nvidia)
    #     vid_stride=1,  # Process every frame
    #     agnostic_nms=False,  # Remove overlapping predictions (if True)
    #     half=True,  # FP16 - 40% faster with minimal accuracy loss
    #     name="football_yolo_v8s_v3",  # folder name to save results
    #     project="runs/train/v8",  # base folder for runs
    #     exist_ok=True,  # overwrite if folder exists
    #     workers=WORKERS,  # parallel data-loading
    #     patience=30,  # Early stopping patience
    #     # scale=0.5,  # Simulates zoom in/out → crucial for tactical view.
    #     # mosaic=1.0,  # Combine 4 images into 1, scaling and flipping them, For ball detection, (small objects) → boosts ball.
    #     # mixup=0.2,  # Blend 2 images together, Improves detection in crowded scenes and color confusion, Reduces misclassification when players overlap.
    #     # hsv_s=0.7,  # Handles: different jerseys, different stadiums, different lighting, yellow referee shirt confusion
    # )

    # model.train(
    #     data="dataset/data.yaml",
    #     imgsz=960,  # larger size = better ball detection
    #     epochs=80,  # best balance for small objects like ball
    #     batch=16,  # -1 = auto-select best batch size
    #     optimizer="auto",  # YOLO learns ideal optimizer
    #     half=True,  # FP16 - 40% faster with minimal accuracy loss
    #     workers=WORKERS,  # parallel data-loading
    #     project="runs/train/v11",
    #     name="football_yolo_v11s_v2",
    #     exist_ok=True,  # overwrite if folder exists
    #     vid_stride=1,  # Process every frame
    #     device=0,  # force use GPU (Nvidia)
    #     # tuning
    #     copy_paste=0.2,  # Copies minority-class objects (ball, referee, goalkeeper) onto images → BOOSTS rare class learning.
    #     mosaic=1.0,  # Combine 4 images into 1, scaling and flipping them, For ball detection, (small objects) → boosts ball.
    #     rect=True,  # Avoids cutting the ball or referee in mosaics.
    #     mixup=0.2,  # Blends images → avoids overfitting on players.
    #     fraction=1.0,  # Ensures each class appears equally often during training.
    # )

    # model.train(
    #     data="dataset/data.yaml",
    #     # Image size strategy:
    #     imgsz=640,  # START HERE - better for small objects
    #     # Training duration:
    #     epochs=80,
    #     # Batch size:
    #     batch=16,  # OK, but reduce to 8 if OOM with imgsz=640
    #     # DISABLE harmful augmentations:
    #     mosaic=0.0,  # TURN OFF COMPLETELY - most important fix!
    #     copy_paste=0.0,  # TURN OFF
    #     mixup=0.0,  # TURN OFF
    #     rect=False,  # Use square training
    #     # Learning:
    #     optimizer="auto",
    #     # Small object specific:
    #     overlap_mask=True,  # Helps small object detection
    #     # Hardware:
    #     half=True,
    #     workers=WORKERS,
    #     device=0,
    #     # Project:
    #     project="runs/train/v11",
    #     name="football_yolo_v11s_v3",
    #     exist_ok=True,
    #     vid_stride=1,
    #     # Monitoring:
    #     patience=50,  # Don't stop early
    # )

    model = YOLO("yolo11s.pt")
    model.train(
        data="dataset_yolo11/data.yaml",
        imgsz=640,  # ±50%
        multi_scale=True,

        epochs=160,
        batch=16,
        device=0,
        half=True,
        workers=WORKERS,
        project="runs/train/v11",
        name="football_yolo_v11s_ball",
        exist_ok=True,
        # chache=True,

        # 🔥 Small / rare object tuning
        overlap_mask=True,
        copy_paste=0.3,
        mosaic=0.0,
        mixup=0.0,

        box=10.0,
        cls=0.5,
        dfl=1.5
    )

    # After training, the best model is saved in:
    # runs/train/football_yolo/weights/best.pt
    print("Training complete!")
