# 🚦 Midad Smart Traffic - Handoff Document

هذا الملف يوثّق حالة المشروع لأي مطور أو AI يريد المساعدة في إكماله.

## 📌 نظرة عامة

**المشروع:** نظام ذكي لعدّ وتصنيف المركبات على طريق مدفوع باستخدام Computer Vision.

**الفريق:**
- Dima Jbara (@DimaMJbara)
- Aya Kabbani (@Ayakabbani)

## 🎯 المهام الموزعة

| المهمة | الملف | المسؤولة | الحالة |
|--------|-------|----------|--------|
| الكشف | detector.py | Dima | ✅ |
| كشف الفيديو | detector_video.py | Dima | ✅ |
| التتبع | tracker.py | Aya | ⏳ |
| العدّ | counter.py | سوا | ⏳ |
| الربط | main.py | سوا | ⏳ |

## 🛠️ التقنيات

- Python 3.14
- YOLO11n (إلزامي)
- OpenCV
- ultralytics
- lap, filterpy (للتتبع)

## 📊 نتائج الكشف

### إعدادات:
python
CONFIDENCE_THRESHOLD = 0.3
CLASS_THRESHOLDS = {
    2: 0.3,      # car
    3: 0.15,     # motorcycle
    5: 0.3,      # bus
    7: 0.3,      # truck
}

معلومات الفيديو:

· الدقة: 1280x720
· FPS: 25
· الإطارات: 7,523
· المدة: 5 دقائق
· الحجم: 487 MB

الكشوفات:

الفئة العدد
car 71,334
motorcycle 23
bus 2,552
truck 7,550

⚠️ الأرقام متكررة (كل مركبة تُكتشف في كل إطار). العدد الفريد بعد tracker.

⚠️ مشاكل معروفة

1. الموتور ضعيف — YOLO11n يكشفه كسيارة. حل مؤقت: عتبة 0.15.
2. الشاحنات الكبيرة — تُكتشف كباص. حل: فلترة بالحجم.
3. العدّ المتكرر — يُحل بـ tracker.

🎯 مهمة Aya: tracker.py

الهدف: إعطاء كل مركبة ID فريد وتتبعها.

المدخلات: نتائج detector.py (bbox + class).

المخرجات: bbox + ID فريد.

الخوارزميات المقترحة:

· IoU Tracker (بسيط)
· Centroid Tracking (متوسط)
· SORT (متقدم)

المكاتب: lap, filterpy

📋 الخطة المتبقية

1. ✅ detector.py
2. ✅ detector_video.py
3. ⏳ tracker.py (Aya)
4. ⏳ counter.py (سوا)
5. ⏳ main.py (سوا)

🔗 روابط

· GitHub: https://github.com/DimaMJbara/midad-smart-traffic-project
· الفيديو: data/raw/traffic_video.mp4

📝 قواعد التعاون

· ممنوع push على main مباشرة
· كل مهمة في فرع منفصل
· Pull Request قبل الدمج

تفاصيل في CONTRIBUTING.md

🚀 للمساعدة من AI خارجي

السياق:

· مشروع مسار تدريبي
· YOLO11n إلزامي
· الهدف: تطوير tracker.py

مهمة Aya:
تطوير tracker.py لـ:

1. إعطاء ID فريد لكل مركبة
2. تتبع المركبة بين الإطارات
3. حل مشكلة العدّ المتكرر

```