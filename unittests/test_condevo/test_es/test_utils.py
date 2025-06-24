from unittest import TestCase, skip


class TestEsUtils(TestCase):
    def test_roulette_wheele(self):
        from condevo.es.utils import roulette_wheel

        import numpy as np
        x = np.linspace(0, 1, 100)
        f = roulette_wheel(x)


        x = np.array([0.1, 2, 2, 2, 3, 3, 4, 4, 2, np.nan, np.nan, 4, 4, 4])
        f = roulette_wheel(x)

        self.assertTrue(len(f) == len(x))
        self.assertTrue(np.all(np.isfinite(f)))
        self.assertTrue(np.all(f[~np.isnan(x)] >= 0))
        self.assertTrue(np.all(f[np.isnan(x)] == 0))
