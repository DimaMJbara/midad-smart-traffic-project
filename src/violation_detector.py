class ViolationDetector:

  def __init__(self, speed_limit=60.0):
    # السرعة القانونية القصوى المسموحة (كم/ساعة)
    self.speed_limit = speed_limit

  def check_speed_violation(self, speed):
    """تتحقق إذا كانت سرعة المركبة تتجاوز السرعة المسموحة"""
    if speed > self.speed_limit:
      return True  # مخالفة تجاوز سرعة
    return False

  def get_box_color(self, is_violation, original_color):
    """إذا كانت مخالفة، نرجع اللون الأحمر (0, 0, 255).

    وإذا لم تكن مخالفة، نحافظ على لونها الأصلي.
    """
    if is_violation:
      return (0, 0, 255)  # أحمر للمخالفة
    return original_color  # اللون الطبيعي الخاص بالمركبة


# --- كود تجريبي لاختبار الفكرة ---
if __name__ == "__main__":
  detector = ViolationDetector(speed_limit=50.0)

  # نفرض أن لدينا سيارتين، لكل واحدة لونها الأصلي (مثلاً أزرق وأصفر)
  car_1_original_color = (255, 0, 0)  # أزرق
  car_2_original_color = (0, 255, 255)  # أصفر

  # اختبار سرعات مختلفة لهاتين السيارتين
  test_cases = [
      {"id": 1, "speed": 45.0, "color": car_1_original_color},  # غير مخالفة
      {"id": 2, "speed": 65.2, "color": car_2_original_color},  # مخالفة سرعة
  ]

  print("--- بدء اختبار تلوين المخالفات مع الحفاظ على الألوان الأصلية ---")
  for case in test_cases:
    is_violating = detector.check_speed_violation(case["speed"])
    final_color = detector.get_box_color(is_violating, case["color"])

    status = (
        "🚨 مخالفة! (صار أحمر)"
        if is_violating
        else "✅ طبيعي (احتفظ بلونه الأصلي)"
    )
    print(
        f"مركبة #{case['id']} | السرعة: {case['speed']} كم/س | الحالة: {status}"
        f" | اللون النهائي (BGR): {final_color}"
    )