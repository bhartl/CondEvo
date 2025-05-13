from unittest import TestCase


class TestFitnessCondition(TestCase):
    def test_charles(self):
        from condevo.es.guidance import FitnessCondition
        condition = FitnessCondition()
