import cv2
from pathlib import Path

# مسار فيديو المرور
VIDEO_PATH = Path("data/raw/traffic_video.mp4")

# مجلد حفظ الصور
OUTPUT_DIR = Path("datasets/vehicle_custom/raw_video_frames")

# استخراج صورة كل ثانيتين
INTERVAL_SECONDS = 2


def main():
    # التأكد من وجود الفيديو
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # فتح الفيديو
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Could not open the video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("Invalid video FPS.")
        cap.release()
        return

    # عدد الإطارات بين كل صورتين محفوظتين
    interval = max(1, round(fps * INTERVAL_SECONDS))

    frame_number = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % interval == 0:
            output_path = (
                OUTPUT_DIR / f"frame_{frame_number:06d}.jpg"
            )

            if cv2.imwrite(str(output_path), frame):
                saved_count += 1

        frame_number += 1

    cap.release()

    print("\n========== Extraction Results ==========")
    print(f"Frames processed: {frame_number}")
    print(f"Images saved: {saved_count}")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()