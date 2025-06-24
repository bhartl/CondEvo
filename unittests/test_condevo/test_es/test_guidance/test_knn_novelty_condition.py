from unittest import TestCase


class TestKNNNoveltyCondition(TestCase):
    def test_charles(self):
        from condevo.es.guidance import KNNNoveltyCondition
        condition = KNNNoveltyCondition()
