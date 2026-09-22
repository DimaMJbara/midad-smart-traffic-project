"""
Vehicle Detector using YOLO11n
كاشف المركبات باستخدام YOLO11n

This module is responsible for:
- Loading the YOLO11n model
- Detecting vehicles in a video frame
- Filtering results to vehicle classes only
"""

import cv2
from ultralytics import YOLO
from pathlib import Path


# ============================================================
# ثوابت المشروع (Project Constants)
# ============================================================

# المسار للفيديو (من جذر المشروع)
VIDEO_PATH = Path("data/raw/traffic_video.mp4")

# مسار حفظ الموديل
MODEL_PATH = "yolo11n.pt"

# الفئات المسموحة في مشروعنا (فئات COCO)
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = {
    2: "car",         # سيارة
    3: "motorcycle",  # موتور
    5: "bus",         # باص
    7: "truck",       # شاحنة
}

# حد الثقة (confidence threshold)
# اخترنا 0.3 بعد تجربة 0.25, 0.3, 0.4, 0.5
# النتيجة: 9 مركبات حقيقية بدون أخطاء
CONFIDENCE_THRESHOLD = 0.3


# ============================================================
# تحميل الموديل (Load Model)
# ============================================================

def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """
    تحميل موديل YOLO11n
    
    Args:
        model_path: مسار ملف الموديل
        
    Returns:
        YOLO model
    """
    print(f"📦 جاري تحميل الموديل: {model_path}")
    model = YOLO(model_path)
    print("✅ تم تحميل الموديل بنجاح")
    return model


# ============================================================
# قراءة إطار من الفيديو (Read a Frame)
# ============================================================

def read_first_frame(video_path: Path):
    """
    قراءة أول إطار من الفيديو
    
    Args:
        video_path: مسار الفيديو
        
    Returns:
        الإطار (frame) أو None إذا فشل
    """
    if not video_path.exists():
        print(f"❌ الفيديو غير موجود: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ فشل فتح الفيديو")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ فشل قراءة الإطار")
        return None
    
    print(f"✅ تم قراءة الإطار بنجاح")
    print(f"📐 الأبعاد: {frame.shape[1]}x{frame.shape[0]} بكسل")
    return frame


# ============================================================
# الكشف (Detection)
# ============================================================

def detect_vehicles(model: YOLO, frame, confidence: float = CONFIDENCE_THRESHOLD):
    """
    كشف المركبات في إطار واحد
    
    Args:
        model: موديل YOLO
        frame: الإطار (numpy array)
        confidence: حد الثقة
        
    Returns:
        نتائج YOLO
    """
    print(f"🔍 جاري الكشف (confidence >= {confidence})...")
    
    results = model(frame, conf=confidence, verbose=False)
    
    return results


# ============================================================
# تصفية النتائج (Filter Results)
# ============================================================

def filter_vehicle_classes(results):
    """
    تصفية النتائج لمركبات فقط
    
    Args:
        results: نتائج YOLO
        
    Returns:
        قائمة (class_id, class_name, confidence, bbox)
    """
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        
        for box in boxes:
            class_id = int(box.cls[0])
            
            # تجاهل الفئات غير المركبات
            if class_id not in VEHICLE_CLASSES:
                continue
            
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                "class_id": class_id,
                "class_name": VEHICLE_CLASSES[class_id],
                "confidence": confidence,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            })
    
    return detections


# ============================================================
# الرسم على الإطار (Draw Detections)
# ============================================================
# ألوان مختلفة لكل فئة
CLASS_COLORS = {
    2: (0, 255, 0),      # car - أخضر
    3: (255, 0, 0),      # motorcycle - أزرق
    5: (0, 165, 255),    # bus - برتقالي
    7: (0, 0, 255),      # truck - أحمر
}


def draw_detections(frame, detections):
    """
    رسم الصناديق والأسماء على الإطار
    
    Args:
        frame: الإطار
        detections: قائمة الكشوفات
        
    Returns:
        الإطار المرسوم عليه
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        class_name = det["class_name"]
        conf = det["confidence"]
        
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        # رسم المستطيل
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # النص (الفئة + الثقة)
        label = f"{class_name} {conf:.2f}"
        
        # خلفية للنص
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        
        # النص
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    
    return frame


# ============================================================
# الدالة الرئيسية (Main)
# ============================================================

def main():
    """الدالة الرئيسية للتشغيل"""
    print("=" * 60)
    print("🚦 نظام كشف المركبات - Midad Smart Traffic Project")
    print("=" * 60)
    
    # 1. قراءة الإطار
    frame = read_first_frame(VIDEO_PATH)
    if frame is None:
        return
    
    # 2. تحميل الموديل
    model = load_model()
    
    # 3. الكشف
    results = detect_vehicles(model, frame)
    
    # 4. التصفية
    detections = filter_vehicle_classes(results)
    print(f"🚗 عدد المركبات المكتشفة: {len(detections)}")
    
    for det in detections:
        print(f"   - {det['class_name']} ({det['confidence']:.2f})")
    
    # 5. الرسم
    frame = draw_detections(frame, detections)
    
    # 6. الحفظ
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "first_frame_detected.jpg"
    
    cv2.imwrite(str(output_path), frame)
    print(f"💾 تم حفظ النتيجة في: {output_path}")
    print("✅ اكتمل التشغيل بنجاح!")


if __name__ == "__main__":
    main()