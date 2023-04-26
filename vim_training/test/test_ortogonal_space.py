import numpy as np
import torch
from sklearn.decomposition import PCA

from vim_training.model import compute_orthogonal_space

DATA = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA()
pca.fit(DATA)
EIG_SPACE = pca.components_


def test_all_components():
    data = torch.from_numpy(DATA)
    eig_space_computed = compute_orthogonal_space(data.float(), 0.0).numpy()

    np.testing.assert_almost_equal(np.absolute(EIG_SPACE), np.absolute(eig_space_computed))


def test_partial():
    data = torch.from_numpy(DATA)
    eig_space_expected = EIG_SPACE.copy()
    eig_space_expected[0, :] = 0.0
    eig_space_computed = compute_orthogonal_space(data.float(), 0.999).numpy()

    np.testing.assert_almost_equal(eig_space_expected, eig_space_computed)


def test_correct_vector_norm():
    data = torch.from_numpy(DATA)
    eig_space_computed = compute_orthogonal_space(data.float(), 0.0).numpy()

    space = EIG_SPACE.copy()
    space[1, :] = 0
    orthogonal_space = EIG_SPACE.copy()
    orthogonal_space[0, :] = 0
    norm_pca = np.matmul(DATA, space.T)
    norm_pca_orthogonal = np.matmul(DATA, orthogonal_space.T)
    
    np.testing.assert_array_less(np.linalg.norm(norm_pca_orthogonal, axis=1), np.linalg.norm(norm_pca, axis=1))
