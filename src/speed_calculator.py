import math


class SpeedCalculator:

  def __init__(self, fps=30, calibration_factor=0.05):
    self.fps = fps
    self.calibration_factor = calibration_factor
    self.previous_positions = {}

  def calculate_speed(self, track_id, current_center):
    """تحسب سرعة المركبة بالـ كم/ساعة بناءً على موقعها الحالي والسابق"""
    if track_id not in self.previous_positions:
      self.previous_positions[track_id] = current_center
      return 0.0

    prev_center = self.previous_positions[track_id]

    # 1. حساب المسافة بالبكسل بين الإطار الحالي والسابق
    pixel_distance = math.sqrt(
        (current_center[0] - prev_center[0]) ** 2
        + (current_center[1] - prev_center[1]) ** 2
    )

    # 2. تحويل المسافة من بكسل إلى أمتار حقيقية
    distance_meters = pixel_distance * self.calibration_factor

    # 3. حساب الزمن المستغرق بين إطارين
    time_seconds = 1.0 / self.fps

    # 4. حساب السرعة بمتر لكل ثانية
    speed_ms = distance_meters / time_seconds

    # 5. التحويل إلى كيلومتر بالساعة
    speed_kmh = speed_ms * 3.6

    # تحديث الموقع القديم
    self.previous_positions[track_id] = current_center

    return round(speed_kmh, 2)


# --- كود التجربة (يجب أن يكون في بداية السطر بدون مسافات بادئة) ---
if __name__ == "__main__":
  calc = SpeedCalculator(fps=30, calibration_factor=0.05)

  fake_positions = [(100, 200), (106, 200), (113, 200), (121, 200)]

  print("--- بدء اختبار حساب السرعة ---")
  for i, pos in enumerate(fake_positions):
    speed = calc.calculate_speed(track_id=1, current_center=pos)
    print(f"الإطار {i+1} - الموقع: {pos} -> السرعة المحسوبة: {speed} كم/ساعة")