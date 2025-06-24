from unittest import TestCase


class TestMLP(TestCase):
    def test_MLP(self):
        from condevo.nn import MLP
        mlp = MLP()
