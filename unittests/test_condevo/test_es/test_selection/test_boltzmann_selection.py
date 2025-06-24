from unittest import TestCase, skip
import torch
import numpy as np
from condevo.es.utils import boltzmann_selection


class TestEsUtils(TestCase):
    @classmethod
    def setUpClass(cls):
        # Set default device for PyTorch for consistency, runs once for the class
        if torch.cuda.is_available():
            torch.set_default_device('cuda')
        else:
            torch.set_default_device('cpu')

    def test_boltzmann_selection(self):
        x = np.linspace(0, 1, 100)
        f = boltzmann_selection(x)

        x = np.array([0.1, 1e5, 1e5, 1e3, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, np.nan, np.nan, 4, 4, 4, 1e6])
        f = boltzmann_selection(np.log(x), consider_nans=False)

        self.assertTrue(len(f) == len(x))
        self.assertTrue(np.all(np.isfinite(f)))
        self.assertTrue(np.all(f[~np.isnan(x)] >= 0))
        self.assertTrue(np.all(f[np.isnan(x)] == 0))

        f1 = boltzmann_selection(np.log(x), consider_nans=True)
        self.assertTrue(not np.all(np.isfinite(f1)))
        nans_indices = np.where(np.isnan(x))[0]
        self.assertTrue(np.all(f[nans_indices] == 0))
        # nans in f1
        self.assertTrue(np.all(np.isnan(f1[nans_indices])))

    def test_basic_functionality_tensor_positive_s(self):
        f = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        result = boltzmann_selection(f, s=3.0, normalize=True)
        self.assertTrue(torch.isclose(result.sum(), torch.tensor(1.0)))
        # Expect higher values to have higher probabilities with positive s
        self.assertGreater(result[4], result[3])
        self.assertGreater(result[3], result[2])
        self.assertGreater(result[2], result[1])
        self.assertGreater(result[1], result[0])

    def test_basic_functionality_numpy_negative_s(self):
        f = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        result = boltzmann_selection(f, s=-0.003, normalize=True) # Negative s favors lower fitness
        self.assertTrue(np.isclose(result.sum(), np.array(1.0)))
        # Expect lower values to have higher probabilities with negative s
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], result[2])
        self.assertGreater(result[2], result[3])
        self.assertGreater(result[3], result[4])
        self.assertIsInstance(result, np.ndarray) # Check return type

    def test_basic_functionality_tensor_no_normalize(self):
        f = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = boltzmann_selection(f, s=1.0, normalize=False)
        # Sum should be approx sum of absolute fitness values
        self.assertTrue(torch.isclose(result.sum(), f.abs().sum()))
        self.assertEqual(result.shape, f.shape)

    def test_empty_input_tensor(self):
        f = torch.tensor([], dtype=torch.float32)
        result = boltzmann_selection(f)
        self.assertEqual(result.numel(), 0)
        self.assertIsInstance(result, torch.Tensor)

    def test_empty_input_numpy(self):
        f = np.array([], dtype=np.float32)
        result = boltzmann_selection(f)
        self.assertEqual(result.size, 0)
        self.assertIsInstance(result, np.ndarray)

    def test_single_element_tensor(self):
        f = torch.tensor([5.0], dtype=torch.float32)
        result = boltzmann_selection(f, normalize=True)
        self.assertTrue(torch.isclose(result.sum(), torch.tensor([1.0])))
        result_no_norm = boltzmann_selection(f, normalize=False)
        self.assertTrue(torch.isclose(result_no_norm.sum(), torch.tensor([5.0])))

    def test_all_same_fitness(self):
        f = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32)
        result = boltzmann_selection(f, s=5.0, normalize=True)
        expected = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)
        self.assertTrue(torch.isclose(result, expected).all())

    def test_all_zero_fitness(self):
        f = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        result = boltzmann_selection(f, s=5.0, normalize=True)
        expected = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)
        self.assertTrue(torch.isclose(result, expected).all())

    def test_s_equals_zero(self):
        f = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = boltzmann_selection(f, s=0.0, normalize=True)
        expected = torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)
        self.assertTrue(torch.isclose(result, expected).all())

    def test_eps_impact(self):
        f = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result_default_eps = boltzmann_selection(f, s=3.0, normalize=True)
        result_larger_eps = boltzmann_selection(f, s=3.0, normalize=True, eps=1e-1)
        self.assertFalse(torch.allclose(result_default_eps, result_larger_eps))

        f_same = torch.tensor([1.0, 1.0], dtype=torch.float32)
        result_same = boltzmann_selection(f_same, eps=1e-12)
        self.assertEqual(result_same.numel(), 2)

    def test_consider_nans_false(self):
        f = torch.tensor([1.0, float('nan'), 3.0, float('inf'), 5.0], dtype=torch.float32)
        result = boltzmann_selection(f, consider_nans=False, normalize=True)
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse(torch.isinf(result).any())
        self.assertTrue(torch.isclose(result.sum(), torch.tensor([1.0])))
        self.assertEqual(result.shape, f.shape)
        # Check if the original NaN/Inf positions got values (they should be 0 from the zeros_like init)
        self.assertTrue(torch.isclose(result[1], torch.tensor([0.0])))
        self.assertTrue(torch.isclose(result[3], torch.tensor([0.0])))

    def test_consider_nans_true(self):
        # check if nans are preserved when consider_nans=True
        f = torch.tensor([1.0, float('nan'), 3.0, float('inf'), 5.0], dtype=torch.float32)
        result = boltzmann_selection(f, consider_nans=True, normalize=True)

        self.assertTrue(torch.isnan(result[1]))
        self.assertTrue(torch.isnan(result[3]))

        # check if rest of values are finite
        self.assertTrue(torch.isfinite(result[0]))
        self.assertTrue(torch.isfinite(result[2]))
        self.assertTrue(torch.isfinite(result[4]))

        # check if sum is still 1
        self.assertTrue(torch.isclose(result[~torch.isnan(result)].sum(), torch.tensor([1.0])))

    def test_threshold_positive(self):
        f = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float32)
        result = boltzmann_selection(f, threshold=0.5, normalize=True)

        self.assertTrue(torch.isclose(result[0], result[1]))
        self.assertTrue(torch.isclose(result[1], result[2]))

        self.assertGreater(result[3], result[0])
        self.assertTrue(torch.isclose(result.sum(), torch.tensor(1.0)))

    def test_threshold_zero(self):
        f = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result_no_threshold = boltzmann_selection(f, threshold=0, normalize=True)
        result_with_zero_threshold = boltzmann_selection(f, threshold=0.0, normalize=True)
        self.assertTrue(torch.isclose(result_no_threshold, result_with_zero_threshold).all())

    def test_threshold_max_value(self):
        f = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float32)
        result = boltzmann_selection(f, threshold=1.0, normalize=True)
        expected = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
        self.assertTrue(torch.isclose(result, expected).all())

    def test_numpy_input_with_nans_and_threshold(self):
        f = np.array([1.0, np.nan, 2.0, -5.0, np.inf, 3.0], dtype=np.float32)
        result = boltzmann_selection(f, s=1.0, normalize=True, consider_nans=True, threshold=0.5)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, f.shape)
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[4]))

        valid_mask = ~np.isnan(result) & ~np.isinf(result)
        valid_results = result[valid_mask]
        self.assertTrue(np.isclose(valid_results.sum(), np.array(1.0)))

        self.assertGreater(result[5], result[2])
        self.assertGreater(result[2], result[0])
        self.assertGreater(result[0], result[3])

    def test_all_nans_or_infs(self):
        f = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=torch.float32)
        result_no_nans = boltzmann_selection(f, consider_nans=False)
        self.assertFalse(torch.any(torch.isnan(result_no_nans)))
        self.assertTrue(torch.isclose(result_no_nans.sum(), torch.tensor(0.0)))
        expected = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.isclose(result_no_nans, expected).all())

        result_with_nans = boltzmann_selection(f, consider_nans=True, normalize=True)
        self.assertTrue(torch.isnan(result_with_nans).all())

    def test_mixed_negative_positive_fitness(self):
        f = torch.tensor([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=torch.float32)
        result = boltzmann_selection(f, s=3.0, normalize=True)
        self.assertTrue(torch.isclose(result.sum(), torch.tensor(1.0)))
        self.assertLess(result[0], result[1])
        self.assertLess(result[1], result[2])
        self.assertLess(result[2], result[3])
        self.assertLess(result[3], result[4])
