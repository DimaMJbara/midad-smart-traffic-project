"""
Smart Traffic System
نظام كشف وتتبع وعدّ المركبات - ملف واحد

This module:
- Detects vehicles using custom trained model (weights.pt)
- Tracks them (assigns unique IDs)
- Counts them using a counting line
- Calculates speed and detects violations with smart coloring
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import time
import numpy as np
import tkinter as tk
from tkinter import ttk

try:
    from .sort import Sort
except ImportError:
    from sort import Sort

# استيراد كلاسات حساب السرعة ورصد المخالفات
from speed_calculator import SpeedCalculator
from violation_detector import ViolationDetector


# ============================================================
# الإعدادات (Configuration)
# ============================================================

VIDEO_PATH = Path("data/raw/traffic_video.mp4")
OUTPUT_PATH = Path("outputs/result_video.mp4")
MODEL_PATH = "models/weights.pt"  # تحديث مسار النموذج الجديد

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

# تعرفة العبور بالدولار حسب توصيف المشروع.
FEE_RATES = {
    "car": 1.0,
    "van": 1.5,
    "bus": 2.0,
    "truck": 2.5,
    "motorcycle": 0.0,                # ممنوع المرور؛ لا تُحتسب رسوم
    "traffic_police_motorcycle": 0.0, # مستثناة من المنع
}

# ألوان الفئات الأصلية (BGR)
CLASS_COLORS = {
    2: (0, 255, 0),      # car - أخضر
    3: (255, 0, 0),      # motorcycle - أزرق
    5: (0, 165, 255),    # bus - أصفر
    7: (255, 0, 255),    # truck - أرجواني
}

# خط العدّ (نسبة من ارتفاع الإطار)
COUNTING_LINE_RATIO = 0.6

# حد تجريبي (None = الفيديو كامل)
MAX_FRAMES = None

FILTER_TYPE = "none"   # "gaussian" / "median" / "none"


# ============================================================
# الفلاتر (Enhancement)
# ============================================================

def enhance_frame(frame, filter_type="median"):
    """تحسين جودة الإطار (إزالة الضوضاء)"""
    if filter_type == "median":
        return cv2.medianBlur(frame, 5)
    elif filter_type == "gaussian":
        return cv2.GaussianBlur(frame, (5, 5), 0)
    else:
        return frame


# ============================================================
# 1. الكشف (Detection)
# ============================================================

def detect_vehicles(model, frame):
    """كشف المركبات وإزالة الصناديق المتداخلة."""
    results = model(frame, conf=0.15, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            conf_val = float(box.conf[0])

             # اطبعي هذا السطر مؤقتاً لنرى ما هي الأرقام التي يخرجها نموذجك الجديد
            print(f"[DEBUG] Detected Class ID: {class_id}, Conf: {conf_val:.2f}")

            if class_id not in VEHICLE_CLASSES:
                continue

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

    # إزالة الصناديق المتداخلة مع الاحتفاظ بالكشف الأعلى ثقة.
    filtered_detections = []
    detections.sort(key=lambda det: det["confidence"], reverse=True)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        should_keep = True

        for kept in filtered_detections:
            kx1, ky1, kx2, ky2 = kept["bbox"]
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_det = max(0, x2 - x1) * max(0, y2 - y1)
            area_kept = max(0, kx2 - kx1) * max(0, ky2 - ky1)
            union = area_det + area_kept - intersection
            iou = intersection / union if union > 0 else 0.0

            if iou > 0.65:
                should_keep = False
                break

        if should_keep:
            filtered_detections.append(det)

    return filtered_detections


def log_detection_diagnostics(frame_no, detections):
    """طباعة تشخيصية للكشف."""
    if detections is None:
        return
    motorcycles = [d for d in detections if d["class_name"] == "motorcycle"]
    buses = [d for d in detections if d["class_name"] == "bus"]
    trucks = [d for d in detections if d["class_name"] == "truck"]

    if motorcycles or buses or trucks:
        print(f"[DETECTION frame={frame_no}] motorcycle={len(motorcycles)}, bus={len(buses)}, truck={len(trucks)}")


# ============================================================
# 2. العدّ (Counting)
# ============================================================

class VehicleCounter:
    """عدّ المركبات حسب النوع واتجاه الحركة."""

    def __init__(self, line_y: int):
        self.line_y = line_y
        self.counts = {name: 0 for name in FEE_RATES}
        self.direction_counts = {
            "toward_camera": {name: 0 for name in FEE_RATES},
            "away_from_camera": {name: 0 for name in FEE_RATES},
        }
        self.revenue_by_class = {name: 0.0 for name in FEE_RATES}
        self.counted_ids = set()
        self.track_history = {}

    def check_crossing(self, det):
        """اكتشاف عبور خط العدّ وتسجيل اتجاه المركبة."""
        if "id" not in det:
            return False

        track_id = det["id"]
        class_name = det["class_name"]

        if class_name not in self.counts or track_id in self.counted_ids:
            return False

        x1, y1, x2, y2 = det["bbox"]
        center_y = (y1 + y2) // 2

        prev_y = self.track_history.get(track_id)
        self.track_history[track_id] = center_y

        if prev_y is None:
            return False

        direction = None
        if prev_y < self.line_y <= center_y:
            direction = "toward_camera"
        elif prev_y > self.line_y >= center_y:
            direction = "away_from_camera"

        if direction is None:
            return False

        self.counts[class_name] += 1
        self.direction_counts[direction][class_name] += 1
        self.revenue_by_class[class_name] += FEE_RATES.get(class_name, 0.0)
        self.counted_ids.add(track_id)
        return True

    def get_total_revenue(self):
        return round(sum(self.revenue_by_class.values()), 2)

    def get_total(self):
        return sum(self.counts.values())


# ============================================================
# 3. الرسم (Drawing مع دعم رصد السرعة والمخالفات)
# ============================================================

def draw_detections(frame, detections, speed_calculator, violation_detector):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        class_name = det["class_name"]
        conf = det["confidence"]
        track_id = det.get("id", "?")
        
        # 1. جلب اللون الأصلي للفئة
        original_color = CLASS_COLORS.get(class_id, (255, 255, 255))
        
        # 2. حساب السرعة الحالية للمركبة إذا توفر معرف التتبع
        speed_text = ""
        final_color = original_color
        
        if track_id != "?":
            current_speed = speed_calculator.calculate_speed(int(track_id), (x1, y1, x2, y2))
            if current_speed is not None:
                speed_text = f" | {current_speed:.1f} km/h"
                # 3. التحقق من وجود مخالفة سرعة وتعديل اللون فقط إن وجدت
                is_violating = violation_detector.check_speed_violation(current_speed)
                final_color = violation_detector.get_box_color(is_violating, original_color)
        
        # رسم المربع باللون المحدد (أصلي أو أحمر عند المخالفة)
        cv2.rectangle(frame, (x1, y1), (x2, y2), final_color, 2)
        
        label = f"ID{track_id} {class_name}{speed_text} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), final_color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def draw_counting_line(frame, line_y):
    h, w = frame.shape[:2]
    color = (255, 255, 0)   # سماوي
    cv2.line(frame, (0, line_y), (w, line_y), color, 3)
    cv2.putText(
        frame, "COUNTING LINE", (10, line_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA
    )
    return frame


# ============================================================
# لوحة الإحصائيات المباشرة
# ============================================================

class LiveStatsDashboard:
    CLASS_LABELS = {
        "car": "سيارة (Car)",
        "van": "فان (Van)",
        "bus": "باص (Bus)",
        "truck": "شاحنة (Truck)",
        "motorcycle": "دراجة نارية (ممنوعة)",
        "traffic_police_motorcycle": "دراجة شرطة مرورية",
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Midad Smart Traffic - Live Statistics")
        self.root.geometry("1040x560")
        self.root.minsize(900, 500)
        self.root.configure(bg="#f3f6fb")
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        tk.Label(self.root, text="MIDAD SMART TRAFFIC", font=("Segoe UI", 18, "bold"), bg="#f3f6fb", fg="#263b63", pady=12).pack()
        tk.Label(self.root, text="لوحة الإحصائيات المباشرة", font=("Segoe UI", 13, "bold"), bg="#f3f6fb", fg="#425779").pack(pady=(0, 12))

        summary = tk.Frame(self.root, bg="#f3f6fb")
        summary.pack(fill="x", padx=18, pady=4)
        self.total_var = tk.StringVar(value="إجمالي المركبات: 0")
        self.frames_var = tk.StringVar(value="الإطارات المعالجة: 0")
        self.fps_var = tk.StringVar(value="سرعة المعالجة: 0.0 FPS")
        self.status_var = tk.StringVar(value="الحالة: جارٍ التحضير...")
        for var in (self.total_var, self.frames_var, self.fps_var, self.status_var):
            tk.Label(summary, textvariable=var, font=("Segoe UI", 10, "bold"), bg="#f3f6fb", fg="#263b63", anchor="w", padx=6, pady=5).pack(side="left", expand=True, fill="x")

        table_frame = tk.Frame(self.root, bg="#f3f6fb")
        table_frame.pack(fill="both", expand=True, padx=20, pady=12)
        columns = ("vehicle", "toward", "away", "total", "rate", "revenue")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=7)
        for col, label, width in (
            ("vehicle", "نوع المركبة", 250),
            ("toward", "ذهاب", 85),
            ("away", "إياب", 85),
            ("total", "العدد الكلي", 95),
            ("rate", "الرسم للمركبة ($)", 130),
            ("revenue", "إجمالي الرسوم ($)", 145),
        ):
            self.table.heading(col, text=label)
            self.table.column(col, width=width, anchor="center")
        
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", font=("Segoe UI", 11), rowheight=34, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#dce6f5")
        self.table.pack(fill="both", expand=True)
        self.revenue_var = tk.StringVar(value="إجمالي رسوم العبور: $0.00")
        tk.Label(self.root, textvariable=self.revenue_var, font=("Segoe UI", 14, "bold"), bg="#f3f6fb", fg="#167345", pady=8).pack()
        self.update_stats(None, 0, 0.0, "بانتظار بدء المعالجة")

    def _on_close(self):
        self.closed = True
        self.root.withdraw()

    def update_stats(self, counter, frame_count, elapsed, status="قيد المعالجة"):
        if self.closed:
            return
        if counter is None:
            counts = {name: 0 for name in self.CLASS_LABELS}
            directions = {"toward_camera": dict(counts), "away_from_camera": dict(counts)}
            revenue_by_class = {name: 0.0 for name in self.CLASS_LABELS}
            total = 0
            total_revenue = 0.0
        else:
            counts = counter.counts
            directions = counter.direction_counts
            revenue_by_class = counter.revenue_by_class
            total = counter.get_total()
            total_revenue = counter.get_total_revenue()

        self.total_var.set(f"إجمالي المركبات: {total}")
        self.frames_var.set(f"الإطارات المعالجة: {frame_count}")
        speed = frame_count / elapsed if elapsed > 0 else 0.0
        self.fps_var.set(f"سرعة المعالجة: {speed:.1f} FPS")
        self.status_var.set(f"الحالة: {status}")
        self.revenue_var.set(f"إجمالي رسوم العبور: ${total_revenue:.2f}")
        for item in self.table.get_children():
            self.table.delete(item)
        for class_name, label in self.CLASS_LABELS.items():
            toward = directions["toward_camera"].get(class_name, 0)
            away = directions["away_from_camera"].get(class_name, 0)
            count = counts.get(class_name, 0)
            rate = FEE_RATES.get(class_name, 0.0)
            revenue = revenue_by_class.get(class_name, 0.0)
            self.table.insert("", "end", values=(label, toward, away, count, f"${rate:.2f}", f"${revenue:.2f}"))
        self.table.insert("", "end", values=("المجموع الكلي", sum(directions["toward_camera"].values()), sum(directions["away_from_camera"].values()), total, "—", f"${total_revenue:.2f}"))
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True


# ============================================================
# 4. الدالة الرئيسية (Main)
# ============================================================

def main():
    print("=" * 60)
    print("🚦 Midad Smart Traffic - Full System")
    print("=" * 60)
    
    print(f"\n[Loading] Model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("[OK] Model loaded successfully!")
    print("Model Classes:", model.names)
    class_name = model.names.get(class_id, "unknown")
    print(f"[DEBUG] Detected Class ID: {class_id} ({class_name}), Conf: {conf_val:.2f}")
    
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video not found: {VIDEO_PATH}")
        return
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("[ERROR] Failed to open video")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.1)

    counting_line_y = int(height * COUNTING_LINE_RATIO)
    counter = VehicleCounter(line_y=counting_line_y)
    
    # تهيئة كلاسات حساب السرعة ورصد المخالفات
    speed_calculator = SpeedCalculator(fps=fps if fps > 0 else 30)
    violation_detector = ViolationDetector(speed_limit=60.0)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))
    
    max_f = MAX_FRAMES if MAX_FRAMES else total_frames
    start_time = time.time()
    frame_count = 0

    try:
        dashboard = LiveStatsDashboard()
    except tk.TclError:
        dashboard = None

    while frame_count < max_f:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_processed = enhance_frame(frame, FILTER_TYPE)
        detections = detect_vehicles(model, frame_processed)

        if len(detections) > 0:
            det_array = np.array([[*d["bbox"], d["confidence"]] for d in detections])
        else:
            det_array = np.empty((0, 5))

        tracks = tracker.update(det_array)

        tracked_detections = []
        used_detection_indices = set()

        for track in tracks:
            x1, y1, x2, y2, track_id = track
            best_iou = 0.0
            best_index = None

            for i, det in enumerate(detections):
                if i in used_detection_indices:
                    continue

                dx1, dy1, dx2, dy2 = det["bbox"]
                
                ix1 = max(x1, dx1)
                iy1 = max(y1, dy1)
                ix2 = min(x2, dx2)
                iy2 = min(y2, dy2)

                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                intersection = iw * ih

                area_track = max(0, x2 - x1) * max(0, y2 - y1)
                area_det = max(0, dx2 - dx1) * max(0, dy2 - dy1)
                union = area_track + area_det - intersection
                iou = intersection / union if union > 0 else 0.0

                if iou > best_iou:
                    best_iou = iou
                    best_index = i

            if best_index is not None and best_iou >= 0.3:
                det = detections[best_index]
                used_detection_indices.add(best_index)

                tracked_detections.append({
                    "id": int(track_id),
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        detections = tracked_detections
        
        for det in detections:
            counter.check_crossing(det)
        
        # تمرير أدوات السرعة والمخالفات لدالة الرسم
        frame = draw_detections(frame, detections, speed_calculator, violation_detector)
        frame = draw_counting_line(frame, counting_line_y)
        
        out.write(frame)
        cv2.imshow("Midad Smart Traffic - Video", frame)
        key = cv2.waitKey(1) & 0xFF
        frame_count += 1

        if dashboard is not None:
            elapsed_now = time.time() - start_time
            dashboard.update_stats(counter, frame_count, elapsed_now)

        if key == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    if dashboard is not None:
        dashboard.update_stats(counter, frame_count, elapsed, "انتهت المعالجة")
        if not dashboard.closed:
            try:
                dashboard.root.mainloop()
            except tk.TclError:
                pass
    
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Output Video: {OUTPUT_PATH}")
    print("Done successfully!")


if __name__ == "__main__":
    main()