"""
Smart Traffic System
نظام كشف وتتبع وعدّ المركبات - ملف واحد

This module:
- Detects vehicles using YOLO11n
- Tracks them (assigns unique IDs)
- Counts them using a counting line
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import time


# ============================================================
# الإعدادات (Configuration)
# ============================================================

VIDEO_PATH = Path("data/raw/traffic_video.mp4")
OUTPUT_PATH = Path("outputs/result_video.mp4")
MODEL_PATH = "yolo11n.pt"

# عتبات الثقة
CONFIDENCE_THRESHOLD = 0.3
CLASS_THRESHOLDS = {
    2: 0.3,      # car
    3: 0.15,     # motorcycle
    5: 0.3,      # bus
    7: 0.3,      # truck
}

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
    5: (0, 165, 255),    # bus - أصفر
    7: (255, 0, 255),      # truck - أرجواني
}

# خط العدّ (نسبة من ارتفاع الإطار)
COUNTING_LINE_RATIO = 0.6

# ⚠️ حد تجريبي (None = الفيديو كامل)
MAX_FRAMES = None


# ============================================================
# 1. الكشف (Detection)
# ============================================================

def detect_vehicles(model, frame):
    """كشف المركبات في إطار واحد"""
    results = model(frame, conf=0.15, verbose=False)
    
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
            threshold = CLASS_THRESHOLDS.get(class_id, 0.3)
            if conf_val < threshold:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                "class_id": class_id,
                "class_name": VEHICLE_CLASSES[class_id],
                "confidence": conf_val,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            })
    
    return detections


# ============================================================
# 2. التتبع (Tracking)
# ============================================================

class SimpleTracker:
    """متتبع بسيط يعطي ID فريد لكل مركبة"""
    
    def __init__(self, max_distance: int = 100):
        self.tracks = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_age = 30
    
    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _distance(self, c1, c2):
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
    
    def update(self, detections, frame_num):
        """تحديث الـ tracks مع الكشوفات الجديدة"""
        
        used_ids = set()
        
        for det in detections:
            det_center = self._center(det["bbox"])
            best_id = None
            
            # ⭐️ معاملة خاصة للموتور
            if det["class_name"] == "motorcycle":
                max_dist = 200       # مسافة أكبر
            else:
                max_dist = self.max_distance
            
            best_distance = max_dist
            
            # ابحثي عن أقرب track مطابق
            for track_id, track in self.tracks.items():
                if track_id in used_ids:
                    continue
                if track["class_name"] != det["class_name"]:
                    continue
                
                track_center = self._center(track["bbox"])
                dist = self._distance(det_center, track_center)
                
                if dist < best_distance:
                    best_distance = dist
                    best_id = track_id
            
            # لو لقينا track → استخدمي الـ ID
            if best_id is not None:
                det["id"] = best_id
                self.tracks[best_id]["bbox"] = det["bbox"]
                self.tracks[best_id]["last_seen"] = frame_num
                used_ids.add(best_id)
            else:
                # أنشئي track جديد
                new_id = self.next_id
                self.next_id += 1
                det["id"] = new_id
                self.tracks[new_id] = {
                    "bbox": det["bbox"],
                    "class_name": det["class_name"],
                    "last_seen": frame_num,
                }
                used_ids.add(new_id)
        
        # ⭐️ احذفي الـ tracks القديمة - معاملة خاصة
        to_delete = []
        for track_id, track in self.tracks.items():
            if track["class_name"] == "motorcycle":
                max_age = 15      # ⭐️ أقل للموتور (كان 30)
            else:
                max_age = self.max_age
            
            if frame_num - track["last_seen"] > max_age:
                to_delete.append(track_id)
        
        for track_id in to_delete:
            del self.tracks[track_id]
        
        return detections



# ============================================================
# 3. العدّ (Counting)
# ============================================================

class VehicleCounter:
    """عدّاد المركبات"""
    
    def __init__(self, line_y: int):
        self.line_y = line_y
        self.counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }
        self.counted_ids = set()
        self.track_history = {}
    
    def check_crossing(self, det):
        if "id" not in det:
            return False
        
        track_id = det["id"]
        
        # ⭐️ الموتور: عدّ فوراً إذا ظهر قريب من الخط
        if det["class_name"] == "motorcycle":
            # إذا عدّيناها، تجاهلي
            if track_id in self.counted_ids:
                return False
            
            # احسبي المسافة من الخط
            x1, y1, x2, y2 = det["bbox"]
            center_y = (y1 + y2) // 2
            distance_from_line = abs(center_y - self.line_y)
            
            # إذا قريب من الخط (200 بكسل) → عدّيه
            if distance_from_line < 200:
                self.counts[det["class_name"]] += 1
                self.counted_ids.add(track_id)
                return True
            
            return False
        
        # للباقي: الطريقة العادية
        if track_id in self.counted_ids:
            return False
        
        x1, y1, x2, y2 = det["bbox"]
        center_y = (y1 + y2) // 2
        
        if track_id in self.track_history:
            prev_y = self.track_history[track_id]
            
            if prev_y < self.line_y <= center_y:
                self.counts[det["class_name"]] += 1
                self.counted_ids.add(track_id)
                return True
        
        self.track_history[track_id] = center_y
        return False

    
    def get_total(self):
        return sum(self.counts.values())


# ============================================================
# 4. الرسم (Drawing)
# ============================================================

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        class_name = det["class_name"]
        conf = det["confidence"]
        track_id = det.get("id", "?")
        
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID{track_id} {class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def draw_counting_line(frame, line_y):
    h, w = frame.shape[:2]
    color = (255, 255, 0)   # 🔵 سماوي (BGR)
    cv2.line(frame, (0, line_y), (w, line_y), color, 3)
    cv2.putText(frame, "COUNTING LINE", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame



def draw_stats(frame, counter):
    y = 30
    cv2.putText(frame, "Counts:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    for cls, count in counter.counts.items():
        y += 30
        text = f"{cls}: {count}"
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y += 40
    total = f"TOTAL: {counter.get_total()}"
    cv2.putText(frame, total, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame


# ============================================================
# 5. الدالة الرئيسية (Main)
# ============================================================

def main():
    print("=" * 60)
    print("🚦 Midad Smart Traffic - Full System")
    print("=" * 60)
    
    print(f"\n📦 تحميل الموديل: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ تم التحميل")
    
    print(f"\n📹 فتح الفيديو: {VIDEO_PATH}")
    if not VIDEO_PATH.exists():
        print(f"❌ الفيديو غير موجود: {VIDEO_PATH}")
        return
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("❌ فشل فتح الفيديو")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   - الأبعاد: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - الإطارات: {total_frames}")
    
    tracker = SimpleTracker()
    counting_line_y = int(height * COUNTING_LINE_RATIO)
    counter = VehicleCounter(line_y=counting_line_y)
    
    print(f"\n🎯 خط العدّ: y = {counting_line_y}")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))
    
    max_f = MAX_FRAMES if MAX_FRAMES else total_frames
    print(f"\n🎬 بدء المعالجة ({max_f} إطار)...\n")
    
    start_time = time.time()
    frame_count = 0
    
    while frame_count < max_f:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detect_vehicles(model, frame)
        detections = tracker.update(detections, frame_count)
        
        for det in detections:
            counter.check_crossing(det)
        
        frame = draw_detections(frame, detections)
        frame = draw_counting_line(frame, counting_line_y)
        frame = draw_stats(frame, counter)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"   ⏳ {frame_count}/{max_f} (⏱️ {elapsed:.0f}s)")
    
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📊 التقرير النهائي")
    print("=" * 60)
    print(f"✅ الإطارات المعالجة: {frame_count}")
    print(f"⏱️  الوقت: {elapsed:.1f} ثانية")
    print(f"🚀 السرعة: {frame_count/elapsed:.1f} إطار/ث")
    print(f"\n🚗 عدّ المركبات:")
    for cls, count in counter.counts.items():
        print(f"   - {cls}: {count}")
    print(f"   ─────────")
    print(f"   الإجمالي: {counter.get_total()}")
    print(f"\n💾 الفيديو الناتج: {OUTPUT_PATH}")
    print("✅ اكتمل التشغيل بنجاح!")


if __name__ == "__main__":
    main()
