from copy import deepcopy
import numpy as np


# to merge areas we need: new weights, new center, new tract lenghts, new region labels -> new conn

# first new voxel count for weighted average
def voxel_count_sum(arr, axis=0, **kwargs):
    """
    Sum voxel over regions to merge. 
    """
    voxel_sum = np.sum(arr[arr > 0], axis=axis)
    print("Voxel count sum: %s" % str(voxel_sum))
    return voxel_sum


def weighted_average(arr, axis=0, **kwargs):
    """
    Finding new weight for merged regions depending on weighted average of weights.
    """
    weights = kwargs.pop('weights', 1.0)
    if weights.ndim < arr.ndim:
        weights = np.expand_dims(weights, 1 - axis)
    assert np.nansum(weights) > 0.0
    w_average = np.nansum(arr * weights, axis=axis, **kwargs) / np.nansum(weights, axis=axis, **kwargs)
    return w_average


def euclidean_distance(p1, p2, mask=None, axis=1):
    return np.sqrt(np.sum(np.square(p1 - p2), axis=axis))


def compute_euclidean_tract_lengths(centres, weights):
    N = centres.shape[0]
    t_length = np.zeros((N, N))
    for iR1 in range(N - 1):
        for iR2 in range(iR1 + 1, N):
            if weights[iR1, iR2] > 0.0:
                t_length[iR1, iR2] = euclidean_distance(centres[iR1][np.newaxis], centres[iR2][np.newaxis])
            else:
                t_length[iR1, iR2] = 0.0
            t_length[iR2, iR1] = t_length[iR1, iR2]
    return t_length


def repeat(arr, sub, axis=0, **kwargs):
    """This function will tile a subarray
       to create an array of shape similar to the input array's arr,
       except for the axis given in the input, where size will be 1.
       It is used to substitute many labels by a single one."""
    shape = list(arr.shape)
    shape[axis] = 1
    return np.tile(sub, tuple(shape))


def merge_axis(inds, arr, axis=0, fun=np.nansum, **funkwargs):
    """This function will merge a subarray of the input array arr,
       as defined by the input indices inds, along the input axis,
       applying the function fun, in order to summarize the values."""
    new_arr = np.delete(arr, inds, axis)
    array_to_be_merged = np.take(arr, inds, axis)
    if funkwargs.get('weights', None) is not None:
        # we need to reduce weights just like arr
        funkwargs['weights'] = np.take(funkwargs['weights'], inds, axis=axis)
    merged_arr = fun(array_to_be_merged, axis, keepdims=True, **funkwargs)
    return insert_axis(new_arr, merged_arr, [np.minimum(inds[0], new_arr.shape[axis])], axis=axis)
    # return np.insert(new_arr, [np.minimum(inds[0], new_arr.shape[axis])], merged_arr, axis=axis)


def merge_nD(inds, arr, fun=np.nansum, weights=None, **funkwargs):
    """This function will merge a subarray of the input array arr,
       as defined by the input indices inds,
       along all the axes of arr (assuming same dimensionality along all axes),
       applying the function fun, in order to summarize the values."""
    new_arr = arr.copy()
    for ax in range(arr.ndim):
        if weights is not None:
            new_arr = merge_axis(inds, new_arr, axis=ax, fun=fun, weights=weights, **funkwargs)
            # we need to reduce weights just like arr
            weights = merge_axis(inds, weights, axis=ax, fun=np.nansum)
        else:
            new_arr = merge_axis(inds, new_arr, axis=ax, fun=fun, **funkwargs)
    return new_arr


def merge_conn(conn, regions, new_label, voxel_count,
               weight_fun=np.nansum, configure=False):
    """This function will merge an input TVB connectivity conn,
       for the input regions (indices or labels),
       substituting them with a summarized region of label new_label,
       applying the summary function for the connectivity weights weight_fun.
       If configure is True, the new connectivity will also be configured."""
    inds = regions
    new_conn = deepcopy(conn)
    repeat_fun = lambda arr, axis=0, **kwargs: repeat(arr, new_label, axis, **kwargs)
    new_conn.centres = merge_axis(inds, conn.centres, axis=0, fun=weighted_average, weights=voxel_count)
    new_conn.region_labels = merge_axis(inds, conn.region_labels, axis=0, fun=repeat_fun)
    new_conn.weights = merge_nD(inds, conn.weights, fun=weight_fun)
    np.fill_diagonal(new_conn.weights, 0.0)
    new_conn.tract_lengths = compute_euclidean_tract_lengths(new_conn.centres, new_conn.weights)
    np.fill_diagonal(new_conn.tract_lengths, 0.0)
    new_conn.tract_lengths[new_conn.weights == 0.0] = 0.0
    if configure:
        new_conn.configure()
    return new_conn


def merge_major_structure(conn, to_merge, struct_labels, voxel_count, weight_fun=np.nansum, configure=False):
    """This function will merge an input TVB connectivity conn,
       for the input major structure label major_struct_to_merge,
       assuming an input vector major_structs_labels, mapping all regions to a major structure,
       substituting merged regions with a summarized region of the major structure label,
       and applying the summary function for the connectivity weights weight_fun.
       If configure is True, the new connectivity will also be configured."""
    regions_inds = np.where([reg_gozzi == to_merge
                             for region, reg_gozzi in zip(conn.region_labels, struct_labels)])[0]
    print("...%d regions' indices of %s:\n%s" % (len(regions_inds), to_merge, str(regions_inds)))
    repeat_fun = lambda arr, axis=0, **kwargs: repeat(arr, to_merge, axis, **kwargs)
    return merge_conn(conn, regions_inds, to_merge, voxel_count,
                      weight_fun=weight_fun, configure=configure), \
        merge_axis(regions_inds, struct_labels, axis=0, fun=repeat_fun), \
        merge_axis(regions_inds, voxel_count, axis=0, fun=voxel_count_sum)


def merge_major_structures(conn, to_merge, merge_labels, voxel_count, weight_fun=np.nansum):
    """This function will merge an input TVB connectivity conn,
       for the input major structures labels major_structs_to_merge,
       assuming an input vector major_structs_labels, mapping all regions to a major structure,
       substituting merged regions with a summarized region of the respective major structure label,
       and applying the summary function for the connectivity weights weight_fun.
       If configure is True, the new connectivity will also be configured."""
    new_conn = deepcopy(conn)
    new_labels = merge_labels.copy()
    new_voxel_count = voxel_count.copy()
    for structures in to_merge:
        print(f"Merging {structures}")
        new_conn, new_labels, new_voxel_count = \
            merge_major_structure(new_conn, structures,
                                  new_labels, new_voxel_count,
                                  weight_fun=weight_fun)

    new_conn.configure()
    return new_conn, new_labels, new_voxel_count
