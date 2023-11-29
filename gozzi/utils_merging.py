import numpy as np


def voxel_count_sum(arr, axis=0, **kwargs):
    """
    Sum voxel over regions to merge. 
    """
    voxel_count_sum = np.sum(arr[arr>0], axis=axis)
    print("Voxel count sum: %s" % str(voxel_count_sum))
    return voxel_count_sum

def weighted_average(arr, axis=0, **kwargs):
    """
    Finding new weight for merged regions depending weighted average of weights. 
    """
    weights = kwargs.pop('weights', 1.0)
    if weights.ndim < arr.ndim:
        weights = np.expand_dims(weights, 1-axis)
    assert np.nansum(weights) > 0.0
    wav = np.nansum(arr * weights, axis=axis, **kwargs) / np.nansum(weights, axis=axis, **kwargs)
    return wav


def resize(arr, sub, axis=0, **kwargs):
    """This function will tile a subarray
       to create an array of shape similar to the input array's arr, 
       except for the axis given in the input, where size will be 1.
       It is used to substitute many labels by a single one."""
    shape = list(arr.shape)
    shape[axis] = 1
    return np.tile(sub, tuple(shape))
