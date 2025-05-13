from unittest import TestCase


class TestCHARLES(TestCase):
    def test_charles(self):
        from condevo.es import CHARLES
        solver = CHARLES(num_params=1)
