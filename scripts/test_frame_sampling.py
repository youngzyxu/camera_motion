import unittest
import importlib.util


class SamplingTest(unittest.TestCase):
    def test_endpoints_and_fractional_fps(self):
        self.assertIsNotNone(importlib.util.find_spec('frame_sampling'), 'Endpoint-safe sampler missing')
        from frame_sampling import sample_indices
        for n, source_fps, fps, expected in [
            (30, 30., 4., [0, 8, 15, 23, 29]),
            (31, 30., 4., [0, 8, 15, 23, 30]),
            (1, 30., 4., [0]),
            (10, 30000/1001, 4., [0, 7, 9]),
            (4, 2., 4., [0, 1, 2, 3]),
        ]:
            with self.subTest(n=n, source_fps=source_fps):
                self.assertEqual(sample_indices(n, source_fps, fps).tolist(), expected)


if __name__ == '__main__':
    unittest.main()
