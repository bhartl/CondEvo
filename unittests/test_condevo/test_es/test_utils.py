from unittest import TestCase


class TestEsUtils(TestCase):
    def test_roulette_wheele(self):
        from condevo.es.utils import roulette_wheel

        import numpy as np
        x = np.linspace(0, 1, 100)
        f = roulette_wheel(x)


        x = np.array([0.1, 2, 2, 2, 3, 3, 4, 4, 2, np.nan, np.nan, 4, 4, 4])
        f = roulette_wheel(x)

        import matplotlib.pyplot as plt
        plt.plot(f)
        plt.show()
