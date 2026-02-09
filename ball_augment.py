import os
import cv2
import random
import shutil

# ----------------------------------------
# SETTINGS
# ----------------------------------------

CLASSES = ['ball', 'goalkeeper', 'player', 'referee']
BALL_CLASS_ID = 0

BALL_CROP_DIR = "dataset/ball_crops"
TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_LABEL_DIR = "dataset/train/labels"

COPIES_PER_IMAGE = (1, 4)  # between 1 and 4 balls per image


# ----------------------------------------
# STEP 1 – Extract all ball crops once
# ----------------------------------------

def extract_ball_crops():
    if os.path.exists(BALL_CROP_DIR):
        shutil.rmtree(BALL_CROP_DIR)
    os.makedirs(BALL_CROP_DIR)

    count = 0

    for label_file in os.listdir(TRAIN_LABEL_DIR):
        if not label_file.endswith(".txt"):
            continue

        img_file = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(TRAIN_IMG_DIR, img_file)
        label_path = os.path.join(TRAIN_LABEL_DIR, label_file)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])
                if cls != BALL_CLASS_ID:
                    continue

                # YOLO format: class cx cy width height
                cx, cy, bw, bh = map(float, parts[1:])
                
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                ball_crop = img[max(0, y1):y2, max(0, x1):x2]
                if ball_crop.size == 0:
                    continue

                save_path = os.path.join(BALL_CROP_DIR, f"ball_{count}.png")
                cv2.imwrite(save_path, ball_crop)
                count += 1

    print(f"[DONE] Extracted {count} ball crops.")


# ----------------------------------------
# STEP 2 – Add random balls to every image
# ----------------------------------------

def add_random_balls_to_dataset():
    ball_files = [f for f in os.listdir(BALL_CROP_DIR) if f.endswith(".png")]
    if len(ball_files) == 0:
        print("No ball crops found. Run extract_ball_crops() first.")
        return

    for img_file in os.listdir(TRAIN_IMG_DIR):
        if not img_file.endswith(".jpg"):
            continue

        img_path = os.path.join(TRAIN_IMG_DIR, img_file)
        label_path = os.path.join(TRAIN_LABEL_DIR, img_file.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(label_path, "a") as label_f:
            # Number of new balls to add
            n_new_balls = random.randint(*COPIES_PER_IMAGE)

            for _ in range(n_new_balls):
                # pick random ball crop
                crop_file = random.choice(ball_files)
                crop = cv2.imread(os.path.join(BALL_CROP_DIR, crop_file))

                if crop is None or crop.size == 0:
                    continue

                ch, cw = crop.shape[:2]

                # random position
                x = random.randint(0, w - cw - 1)
                y = random.randint(0, h - ch - 1)

                # Copy-paste with simple blending
                roi = img[y:y + ch, x:x + cw]
                blended = cv2.addWeighted(roi, 0.3, crop, 0.7, 0)
                img[y:y + ch, x:x + cw] = blended

                # Convert to YOLO label
                cx = (x + cw / 2) / w
                cy = (y + ch / 2) / h
                bw = cw / w
                bh = ch / h

                # add line to label file
                label_f.write(f"\n{BALL_CLASS_ID} {cx} {cy} {bw} {bh}")

        # overwrite modified image
        cv2.imwrite(img_path, img)

    print("[DONE] Augmented dataset with synthetic balls.")


# ----------------------------------------
# MAIN
# ----------------------------------------

if __name__ == "__main__":
    print("Extracting ball crops...")
    extract_ball_crops()

    print("Adding synthetic balls to dataset...")
    add_random_balls_to_dataset()

    print("All done!")
