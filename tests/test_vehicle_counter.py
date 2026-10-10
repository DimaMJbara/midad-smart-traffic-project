
import unittest
from src.main import VehicleCounter


def make_detection(track_id, class_name, center_y):
    return {
        "id": track_id,
        "class_name": class_name,
        "bbox": (100, center_y - 10, 150, center_y + 10),
    }


class TestVehicleCounter(unittest.TestCase):

    def test_count_crossing_once(self):
        counter = VehicleCounter(line_y=100)

        # أول ظهور قبل الخط
        self.assertFalse(
            counter.check_crossing(make_detection(1, "car", 90))
        )

        # عبور الخط من الأعلى للأسفل
        self.assertTrue(
            counter.check_crossing(make_detection(1, "car", 110))
        )

        # استمرار الحركة بعد العبور يجب ألا يكرر العدّ
        self.assertFalse(
            counter.check_crossing(make_detection(1, "car", 120))
        )

        self.assertEqual(counter.counts["car"], 1)
        self.assertEqual(counter.get_total(), 1)

    def test_direction_counts(self):
        counter = VehicleCounter(line_y=100)

        counter.check_crossing(make_detection(1, "car", 90))
        counter.check_crossing(make_detection(1, "car", 110))

        counter.check_crossing(make_detection(2, "truck", 110))
        counter.check_crossing(make_detection(2, "truck", 90))

        self.assertEqual(
            counter.get_direction_total("toward_camera"), 1
        )
        self.assertEqual(
            counter.get_direction_total("away_from_camera"), 1
        )

    def test_unknown_vehicle_is_not_counted(self):
        counter = VehicleCounter(line_y=100)

        counter.check_crossing(make_detection(1, "bicycle", 90))
        result = counter.check_crossing(
            make_detection(1, "bicycle", 110)
        )

        self.assertFalse(result)
        self.assertEqual(counter.get_total(), 0)


if __name__ == "__main__":
    unittest.main()
