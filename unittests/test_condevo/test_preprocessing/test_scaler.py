import unittest
import torch

from condevo.preprocessing import Scaler
from condevo.preprocessing import StandardScaler


class TestScalerIdentity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_identity_no_nans_no_conditions_default_weights(self):
        scaler = Scaler()

        x = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]])
        z, w, conds = scaler.fit_transform(x, weights=None, conditions=())

        self.assertTrue(torch.equal(z, x))
        self.assertEqual(tuple(conds), ())
        self.assertEqual(tuple(w.shape), (x.shape[0], 1))
        self.assertTrue(torch.allclose(w, torch.ones_like(w)))

    def test_identity_drops_nan_in_x_no_conditions(self):
        scaler = Scaler()

        x = torch.tensor([[1.0, 2.0],
                          [float("nan"), 4.0],
                          [5.0, 6.0]])
        weights = torch.ones((3, 1))

        z, w, conds = scaler.fit_transform(x, weights=weights, conditions=())

        expected = torch.tensor([[1.0, 2.0],
                                 [5.0, 6.0]])
        self.assertTrue(torch.equal(z, expected))
        self.assertEqual(w.shape[0], 2)
        self.assertEqual(tuple(conds), ())

    def test_identity_drops_nan_in_weights_no_conditions(self):
        scaler = Scaler()

        x = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0],
                          [5.0, 6.0]])
        weights = torch.tensor([[1.0],
                                [float("nan")],
                                [1.0]])

        z, w, conds = scaler.fit_transform(x, weights=weights, conditions=())

        expected = torch.tensor([[1.0, 2.0],
                                 [5.0, 6.0]])
        self.assertTrue(torch.equal(z, expected))
        self.assertEqual(w.shape[0], 2)
        self.assertTrue(torch.all(torch.isfinite(w)))
        self.assertEqual(tuple(conds), ())

    def test_identity_nan_in_x_filters_conditions(self):
        scaler = Scaler()

        x = torch.tensor([[1.0, 2.0],
                          [float("nan"), 4.0],
                          [5.0, 6.0]])
        weights = torch.ones((3, 1))
        c1 = torch.tensor([[10.0],
                           [20.0],
                           [30.0]])
        c2 = torch.tensor([[1.0, 1.0],
                           [2.0, 2.0],
                           [3.0, 3.0]])

        z, w, conds = scaler.fit_transform(x, weights=weights, conditions=(c1, c2))

        expected_x = torch.tensor([[1.0, 2.0],
                                   [5.0, 6.0]])
        expected_c1 = torch.tensor([[10.0],
                                    [30.0]])
        expected_c2 = torch.tensor([[1.0, 1.0],
                                    [3.0, 3.0]])

        self.assertTrue(torch.equal(z, expected_x))
        self.assertTrue(torch.equal(conds[0], expected_c1))
        self.assertTrue(torch.equal(conds[1], expected_c2))
        self.assertEqual(w.shape[0], 2)


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_fit_sets_mean_std_shapes(self):
        scaler = StandardScaler()

        x = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0],
                          [5.0, 6.0]])
        z, w, conds = scaler.fit_transform(x, weights=None, conditions=())

        self.assertEqual(tuple(scaler.mean.shape), (1, 2))
        self.assertEqual(tuple(scaler.std.shape), (1, 2))
        self.assertEqual(tuple(z.shape), tuple(x.shape))
        self.assertEqual(tuple(conds), ())
        self.assertEqual(w.shape[0], x.shape[0])

    def test_transform_inverse_roundtrip(self):
        scaler = StandardScaler()

        x = torch.randn(20, 4)
        z, _, _ = scaler.fit_transform(x, weights=None, conditions=())
        x_rec = scaler.inverse_transform(z)

        self.assertTrue(torch.allclose(x_rec, x, atol=1e-6, rtol=1e-6))

    def test_transform_zero_mean_unit_std_unweighted(self):
        scaler = StandardScaler()

        x = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0],
                          [5.0, 6.0]])
        z, _, _ = scaler.fit_transform(x, weights=None, conditions=())

        z_mean = z.mean(dim=0)
        z_std = z.std(dim=0, unbiased=False)

        self.assertTrue(torch.allclose(z_mean, torch.zeros_like(z_mean), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(z_std, torch.ones_like(z_std), atol=1e-6, rtol=1e-6))

    def test_nan_in_x_drops_rows_and_conditions_and_affects_stats(self):
        scaler = StandardScaler()

        x = torch.tensor([[1.0, 2.0],
                          [float("nan"), 4.0],
                          [5.0, 6.0]])
        weights = torch.ones((3, 1))
        c = torch.tensor([[10.0],
                          [20.0],
                          [30.0]])

        z, w, conds = scaler.fit_transform(x, weights=weights, conditions=(c,))

        self.assertEqual(z.shape[0], 2)
        self.assertEqual(w.shape[0], 2)
        self.assertTrue(torch.equal(conds[0], torch.tensor([[10.0], [30.0]])))

        # remaining x: [[1,2],[5,6]] mean = [3,4]
        self.assertTrue(torch.allclose(scaler.mean, torch.tensor([[3.0, 4.0]]), atol=1e-6, rtol=1e-6))

    def test_nan_in_weights_drops_rows_and_conditions_and_affects_stats(self):
        scaler = StandardScaler()

        x = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0],
                          [5.0, 6.0]])
        weights = torch.tensor([[1.0],
                                [float("nan")],
                                [1.0]])
        c = torch.tensor([[10.0],
                          [20.0],
                          [30.0]])

        z, w, conds = scaler.fit_transform(x, weights=weights, conditions=(c,))

        self.assertEqual(z.shape[0], 2)
        self.assertEqual(w.shape[0], 2)
        self.assertTrue(torch.equal(conds[0], torch.tensor([[10.0], [30.0]])))

        # remaining x: [[1,2],[5,6]] mean = [3,4]
        self.assertTrue(torch.allclose(scaler.mean, torch.tensor([[3.0, 4.0]]), atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
