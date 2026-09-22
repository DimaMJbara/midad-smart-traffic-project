# دليل المساهمة

## قواعد الفريق

### 1. ممنوع push مباشر على main
كل تعديل لازم يمر عبر فرع (branch) منفصل.

### 2. كل مهمة في فرع خاص
- للكشف: `feature-detector`
- للتتبع: `feature-tracker`
- للعدّ: `feature-counter`

### 3. رسائل commit واضحة
- ❌ `تعديل`
- ✅ `Add vehicle detection logic in detector.py`

### 4. مراجعة الكود قبل الدمج
افتحي Pull Request دائماً.

## خطوات العمل اليومية

### قبل ما تبدأي:
git checkout main
git pull

### افتحي فرع جديد:
git checkout -b feature-اسم-المهمة

### بعد ما تخلصي:
git add .
git commit -m "وصف واضح"
git push origin feature-اسم-المهمة

### بعدها:
افتحي Pull Request على GitHub.

## تقسيم المهام

| المسؤولة | الملفات |
|----------|---------|
| Dima | src/detector.py, src/counter.py |
| Aya | src/tracker.py, src/toll_calculator.py |
| الاثنتان | src/main.py, README.md |

## التواصل

- **WhatsApp:** للأسئلة السريعة
- **GitHub Issues:** للمهام الرسمية
- **Pull Requests:** للمراجعات
