# Midad Smart Traffic Project

نظام ذكي لعدّ وتصنيف المركبات على طريق مدفوع باستخدام Computer Vision.

## نظرة عامة

يهدف المشروع إلى تطوير نظام ذكي يُثبّت على كاميرا مراقبة فوق طريق مدفوع (Toll Road) ذي اتجاهين، بحيث يقوم النظام بـ:

- كشف المركبات وتصنيفها (سيارة، فان، باص، شاحنة، موتور)
- عدّ المركبات المارة مع منع الازدواج
- حساب رسوم العبور حسب نوع المركبة
- مراقبة السرعة والإبلاغ عن المخالفات
- إصدار تقارير نهائية بالإيرادات

## التقنيات المستخدمة

- Python 3.10+
- YOLO11n (Ultralytics) - لكشف المركبات
- OpenCV - لمعالجة الفيديو
- IoU/Centroid Tracking - لتتبع المركبات
- NumPy, Matplotlib - للمعالجة والرسم

## هيكل المشروع

midad-smart-traffic-project/
- src/              الكود الأساسي
- notebooks/        تجارب Jupyter
- data/             البيانات (غير مرفوعة)
- models/           موديلات YOLO (غير مرفوعة)
- outputs/          نتائج التشغيل
- configs/          ملفات الإعدادات
- docs/             التوثيق
- tests/            اختبارات

## التثبيت

### المتطلبات
- Python 3.10 أو أحدث
- Git

### الخطوات

git clone https://github.com/DimaMJbara/midad-smart-traffic-project.git
cd midad-smart-traffic-project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

## الفريق

- Dima Jbara - @DimaMJbara
- Aya Kabbani - @Ayakabbani

## الرخصة

هذا المشروع مرخص تحت MIT License.
