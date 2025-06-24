from unittest import TestCase


class TestHADES(TestCase):
    def test_hades(self):
        from condevo.es import HADES
        solver = HADES(num_params=1)
