"""
Video Detector using YOLO11n
كاشف المركبات على الفيديو باستخدام YOLO11n

This module processes an entire video:
- Reads each frame
- Detects vehicles
- Draws boxes
- Saves output video
- Prints final statistics
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import time


# ============================================================
# الثوابت (Constants)
# ============================================================

VIDEO_PATH = Path("data/raw/traffic_video.mp4")
OUTPUT_PATH = Path("outputs/detected_video.mp4")
MODEL_PATH = "yolo11n.pt"

# ✅ العتبة النهائية (بعد التجارب)
CONFIDENCE_THRESHOLD = 0.3

# فئات المركبات
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ألوان الفئات (BGR)
CLASS_COLORS = {
    2: (0, 255, 0),      # car - أخضر
    3: (255, 0, 0),      # motorcycle - أزرق
    5: (0, 165, 255),    # bus - برتقالي
    7: (0, 0, 255),      # truck - أحمر
}

# ⚠️ حد تجريبي: كم إطار نعالج؟
# None = الفيديو كامل
# رقم = عدد الإطارات (مثلاً 300 = 10 ثواني)
MAX_FRAMES = 300


# ============================================================
# تحميل الموديل
# ============================================================

def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """تحميل موديل YOLO11n"""
    print(f"📦 جاري تحميل الموديل: {model_path}")
    model = YOLO(model_path)
    print("✅ تم تحميل الموديل بنجاح")
    return model


# ============================================================
# الرسم على الإطار
# ============================================================

def draw_detections(frame, detections):
    """رسم الصناديق والأسماء على الإطار"""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        class_name = det["class_name"]
        conf = det["confidence"]
        
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        # المستطيل
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # النص
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


# ============================================================
# معالجة الفيديو (الدالة الرئيسية)
# ============================================================

def process_video(video_path: Path, output_path: Path, model: YOLO,
                  confidence: float = CONFIDENCE_THRESHOLD,
                  max_frames: int = None):
    """
    معالجة فيديو كامل: كشف + رسم + حفظ
    
    Args:
        video_path: مسار الفيديو الأصلي
        output_path: مسار الفيديو الناتج
        model: موديل YOLO
        confidence: عتبة الثقة
        max_frames: أقصى عدد إطارات (None = الكل)
    
    Returns:
        dict فيه إحصائيات
    """
    
    # 1. فتح الفيديو الأصلي
    if not video_path.exists():
        print(f"❌ الفيديو غير موجود: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("❌ فشل فتح الفيديو")
        return None
    
    # 2. معلومات الفيديو
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 معلومات الفيديو:")
    print(f"   - الأبعاد: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - عدد الإطارات: {total_frames}")
    print(f"   - المدة: {total_frames/fps:.1f} ثانية")
# 3. إذا max_frames محدد، نستخدمه
    if max_frames is None:
        max_frames = total_frames
    
    print(f"\n🎬 بدء المعالجة... (عدد الإطارات: {max_frames})")
    print(f"🔍 العتبة: {confidence}")
    
    # 4. تحضير فيديو الإخراج
    output_path.parent.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # 5. إحصائيات
    frame_count = 0
    total_vehicles = 0
    class_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    start_time = time.time()
    
    # 6. المعالجة
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # الكشف
        results = model(frame, conf=confidence, verbose=False)
        
        # التصفية
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASSES:
                    continue
                
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "class_id": class_id,
                    "class_name": VEHICLE_CLASSES[class_id],
                    "confidence": conf_val,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })
        
        # الرسم
        frame = draw_detections(frame, detections)
        
        # حفظ الإطار
        out.write(frame)
        
        # إحصائيات
        frame_count += 1
        total_vehicles += len(detections)
        for det in detections:
            class_counts[det["class_name"]] += 1
        
        # تقدم كل 100 إطار
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            speed = frame_count / elapsed
            remaining = (max_frames - frame_count) / speed if speed > 0 else 0
            print(f"   ⏳ {frame_count}/{max_frames} إطار "
                  f"(سرعة: {speed:.1f} إطار/ث، متبقي: {remaining:.0f} ث)")
    
    # 7. إنهاء
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    
    # 8. الإحصائيات النهائية
    stats = {
        "frames_processed": frame_count,
        "total_vehicles_detected": total_vehicles,
        "class_counts": class_counts,
        "time_seconds": elapsed,
        "fps_processing": frame_count / elapsed if elapsed > 0 else 0,
    }
    
    return stats


# ============================================================
# الدالة الرئيسية
# ============================================================

def main():
    """الدالة الرئيسية"""
    print("=" * 60)
    print("🚦 معالجة الفيديو - Midad Smart Traffic Project")
    print("=" * 60)
    
    # تحميل الموديل
    model = load_model()
    
    # معالجة الفيديو
    stats = process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        model=model,
        confidence=CONFIDENCE_THRESHOLD,
        max_frames=MAX_FRAMES,
    )
    
    if stats is None:
        return
    
    # طباعة الإحصائيات
    print("\n" + "=" * 60)
    print("📊 الإحصائيات النهائية")
    print("=" * 60)
    print(f"✅ عدد الإطارات المعالجة: {stats['frames_processed']}")
    print(f"⏱️  الوقت المستغرق: {stats['time_seconds']:.1f} ثانية")
    print(f"🚀 سرعة المعالجة: {stats['fps_processing']:.1f} إطار/ث")
    print(f"🚗 إجمالي المركبات المكتشفة: {stats['total_vehicles_detected']}")
    print(f"\n📋 حسب الفئة:")
    for cls, count in stats["class_counts"].items():
        print(f"   - {cls}: {count}")
    
    print(f"\n💾 الفيديو الناتج: {OUTPUT_PATH}")
    print("✅ اكتمل التشغيل بنجاح!")


if __name__ == "__main__":
    main()
