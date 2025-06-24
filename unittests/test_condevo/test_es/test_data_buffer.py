from unittest import TestCase


class TestDataBuffer(TestCase):
    def test_databuffer(self):
        from condevo.es.data import DataBuffer
        data_buffer = DataBuffer(max_size=1)
