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

    # 1️⃣ Load your previously trained model
    # This could be your best.pt or last.pt from previous training
    model = YOLO("runs/train/v11/football_yolo_v11s_ball/weights/best.pt")
    # model.info()

    # 2️⃣ Fine-tune on the new dataset
    # Make sure your YAML file points to the new dataset
    # and has correct classes
    model.train(
        data="Ghassan Dataset/data.yaml",  # new dataset YAML
        imgsz=640,  # ±50%
        multi_scale=True,

        epochs=40,
        batch=16,
        device=0,
        half=True,
        workers=WORKERS,
        project="runs/train/v11",
        name="football_yolo_v11s_ball_finetuned",
        exist_ok=True,

        lr0=0.001,  # initial learning rate
        lrf=0.01,  # final LR decay factor
        freeze=11,  # freeze backbone
    )

    # 3️⃣ Evaluate on validation set (optional)
    metrics = model.val()
    print(metrics)

    # 4️⃣ fine-tuned model
    print("Training complete!")
