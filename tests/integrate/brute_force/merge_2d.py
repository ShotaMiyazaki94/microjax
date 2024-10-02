import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

def detect_overlaps(regions):
    """
    Detect overlapping regions in a given set of radial and angular boundaries.

    This function takes an array of regions defined by their radial (r) and 
    angular (theta) boundaries and returns a matrix indicating which regions overlap.

    Each region is described by four values:
    - r_min: the minimum radial boundary
    - r_max: the maximum radial boundary
    - t_min: the minimum angular boundary (theta)
    - t_max: the maximum angular boundary (theta)

    Parameters
    ----------
    regions : numpy.ndarray
        A 2D array of shape (N, 4), where N is the number of regions.
        Each row represents a region with the following structure:
        [r_min, r_max, t_min, t_max], where:
        - r_min (float): minimum radial boundary
        - r_max (float): maximum radial boundary
        - t_min (float): minimum angular boundary (theta)
        - t_max (float): maximum angular boundary (theta)

    Returns
    -------
    overlap_matrix : numpy.ndarray
        A boolean 2D array of shape (N, N), where the element at position (i, j)
        is True if region i overlaps with region j in both radial and angular boundaries,
        and False otherwise. The matrix is symmetric, and the diagonal elements represent
        self-overlap (which will always be True).

    Notes
    -----
    - The overlap in radial and angular directions is calculated separately, and two 
      regions are considered overlapping if they overlap in both the radial and angular 
      boundaries.
    - The method compares each region against every other region in the array.

    Example
    -------
    >>> regions = np.array([[0.0, 1.0, 0.0, np.pi/4],
                            [0.5, 1.5, np.pi/6, np.pi/2],
                            [2.0, 3.0, np.pi/3, 2*np.pi/3]])
    >>> overlaps = detect_overlaps(regions)
    >>> print(overlaps)
    [[ True  True False]
     [ True  True False]
     [False False  True]]
    """
    r_min = regions[:, 0]
    r_max = regions[:, 1]
    t_min = regions[:, 2]
    t_max = regions[:, 3]

    r_min_exp = r_min[:, None]
    r_max_exp = r_max[:, None]
    t_min_exp = t_min[:, None]
    t_max_exp = t_max[:, None]

    r_overlap = ~(r_max[:, None] < r_min_exp.T) & ~(r_max_exp.T < r_min[:, None])
    t_overlap = ~(t_max[:, None] < t_min_exp.T) & ~(t_max_exp.T < t_min[:, None])

    overlap_matrix = r_overlap & t_overlap
    return overlap_matrix

def merge_overlapping_regions(regions):
    overlap_matrix = detect_overlaps(regions)
    N = regions.shape[0]
    overlap_matrix = overlap_matrix | overlap_matrix.T  # Ensure symmetry

    # Initialize labels for each region
    labels = jnp.arange(N)

    def propagate_labels(labels):
        def update_labels(i, labels):
            overlaps = overlap_matrix[i]
            min_label = jnp.min(jnp.where(overlaps, labels, N))
            labels = jnp.where(overlaps, min_label, labels)
            return labels

        # Apply label updates for all regions
        labels = jax.lax.fori_loop(0, N, update_labels, labels)
        return labels

    # Iterate label propagation until convergence
    def cond_fun(carry):
        old_labels, new_labels = carry
        return jnp.any(old_labels != new_labels)

    def body_fun(carry):
        _, labels = carry
        new_labels = propagate_labels(labels)
        return labels, new_labels

    initial_labels = labels
    _, final_labels = jax.lax.while_loop(
        cond_fun, body_fun, (initial_labels - 1, initial_labels)
    )

    unique_labels = jnp.unique(final_labels)

    def merge_group(label):
        mask = final_labels == label
        regions_to_merge = jnp.where(mask[:, None], regions, jnp.array([jnp.inf, -jnp.inf, jnp.inf, -jnp.inf]))
        r_min = jnp.min(regions_to_merge[:, 0])
        r_max = jnp.max(regions_to_merge[:, 1])
        t_min = jnp.min(regions_to_merge[:, 2])
        t_max = jnp.max(regions_to_merge[:, 3])
        return jnp.array([r_min, r_max, t_min, t_max])

    # Merge regions for each unique label
    merged_regions_per_label = jax.vmap(merge_group)(unique_labels)

    # Map merged regions back to the positions of the original regions
    def get_merged_region(region_label):
        # Create a mask over unique_labels
        mask = region_label == unique_labels
        # Use the mask to select the merged region
        merged_region = jnp.sum(jnp.where(mask[:, None], merged_regions_per_label, 0.0), axis=0)
        return merged_region

    merged_regions = jax.vmap(get_merged_region)(final_labels)

    # Create a mask indicating unique regions (first occurrence of each label)
    _, index = jnp.unique(final_labels, return_index=True)
    mask = jnp.zeros(N, dtype=bool).at[index].set(True)

    return merged_regions, mask



regions = jnp.array([[0.0, 1.0, 0.0, 0.1],
                     [0.9, 1.3, 0.0, 0.1],
                     [10, 11, 0, 2*jnp.pi]
])
merged_regions = jax.jit(merge_overlapping_regions)(regions)
print(regions)
print("Merged Regions:")
print(merged_regions)