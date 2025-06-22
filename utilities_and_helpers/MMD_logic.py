from itertools import pairwise
from multiprocessing import Pool
import multiprocessing

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

'''
MMD implementation

This implementation of the Maximum Mean Discrepancy (MMD) test is based on the code from Roschewitz et al. (2024) (https://github.com/biomedia-mira/shift_identification)
It has been adapted to work directly with dataloaders and to use PCA for dimensionality reduction before computing the MMD statistic.
Contrary to the original code, this implementation does not use a fixed number of samples for the MMD test, but instead uses all available samples from the dataloaders.
Additionally, it returns both the MMD statistic and the p-value for each scanner pair.

explanation of permutation test:
Null hypothesis: The two distributions are the same.
We shuffle the combined samples and compute the MMD statistic for each permutation.
if X and Y are from the same distribution, the MMD statistic should be close to 0 even if we shuffle the samples.
if X and Y are from different distributions, the MMD statistic should be large after shuffling.
The p-value is the proportion of permuted MMD statistics that are greater than the observed MMD statistic.
If the p-value is less than a significance level (e.g., 0.05), we reject the null hypothesis. -> the two distributions are different.
'''


def permutation_worker(p, n1, rbf_distances, observed):
    """
    Wrapper function to run multiprocessing permutation test.
    """
    null_observed = get_mmd_from_all_distances(rbf_distances[p][:, p], n1)
    return 1 if null_observed >= observed else 0

def get_mmd_from_all_distances(distances, n1):
    """
    Computes MMD estimator as per Gretton et al. `A Kernel Two-Sample Test`.
    """
    XX = distances[:n1, :n1]
    YY = distances[n1:, n1:]
    XY = distances[:n1, n1:]
    n2 = distances.shape[0] - n1
    return (
        (XX.sum() - np.trace(XX)) / (n1**2 - n1)
        + (YY.sum() - np.trace(YY)) / (n2**2 - n2)
        - 2 * XY.mean()
    )

def mmd_permutation_test(test_loaders, scanner, scan, perms):
    """
    Perform MMD permutation test on PCA-transformed test data with parallelization.

    Parameters:
        test_loaders (dict): Dictionary of test loaders for each scanner.
        scanner (str): The scanner to compare against others.
        scan (list): List of all scanners.
        perms (int): Number of permutations for the test.

    Returns:
        mmd_results (dict): MMD statistics for each scanner pair.
        p_values (dict): p-values for each scanner pair.
    """
    mmd_results = {}
    p_values = {}

    # Extract data for the selected scanner
    data_A = []
    for patches, _ in test_loaders[scanner]:
        data_A.append(patches.numpy())
    data_A = np.vstack(data_A)

    for scan_item in scan:
        # Extract data for the other scanner
        data_B = []
        for patches, _ in test_loaders[scan_item]:
            data_B.append(patches.numpy())
        data_B = np.vstack(data_B)

        # Apply PCA with 32 components
        pca = PCA(n_components=32)
        combined_data = np.vstack((data_A, data_B))
        combined_data_pca = pca.fit_transform(combined_data)
        data_A_pca = combined_data_pca[:len(data_A)]
        data_B_pca = combined_data_pca[len(data_A):]

        # Compute pairwise distances and RBF kernel
        n1 = len(data_A_pca)
        distances = pairwise_distances(combined_data_pca, n_jobs=multiprocessing.cpu_count() - 1)
        sigma = np.median(distances)
        gamma = 1 / sigma
        rbf_distances = pairwise.rbf_kernel(combined_data_pca, combined_data_pca, gamma)

        # Observed MMD
        observed_mmd = get_mmd_from_all_distances(rbf_distances, n1)

        # Parallelized permutation test
        with Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            results = pool.starmap(
                permutation_worker,
                [
                    (np.random.permutation(len(combined_data_pca)), n1, rbf_distances, observed_mmd)
                    for _ in range(perms)
                ],
            )
            larger = sum(results) + 1

        p_value = larger / perms
        # print(f"MMD Statistic between {scanner} and {scan_item}: {observed_mmd:.10f}, p-value: {p_value:.10f}")
        mmd_results[f'{scanner}_{scan_item}'] = observed_mmd
        p_values[f'{scanner}_{scan_item}'] = p_value

    return mmd_results, p_values