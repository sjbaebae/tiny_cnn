import numpy as np
import models

# TODO: implement conv2d and pooling layers

def _im2col_col2im_indices_along_dim(input_size, kernel_size, stride, padding, dilation):
    """
    Helper function to compute the indices for im2col and col2im operations.
    """
    # want to find max possible block_d along dim. So remove (true_kernel_size - 1) from right end
    blocks_d = input_size + 2 * padding - dilation * (kernel_size - 1)

    # can use output_size to limit the range of indices

    # calculate all start positions for blocks. 
    # broadcast on 0th dim, hence unsqueeze on 0th
    block_indices = np.arange(0, blocks_d, stride)[None, :] # shape (1, n_blocks)

    # now compute deltas from these start positions, moving by dilation. Since cant reach kernel_size * dilation, best is (kernel_size - 1) * dilation, since we move by dilation
    # boradcast on -1th dim, hence unsqueeze on -1th
    kernel_deltas = np.arange(0, kernel_size * dilation, dilation)[:, None] # shape (kernel_size, 1)

    # now we can broadcast to get the true indices along the dim. 
    # Note for our purposes, will follow the convention of (kernel_size, n_blocks). Hence broadcast, blocks on 0th dim, kernel on 1st dim

    return block_indices + kernel_deltas # shape (kernel_size, n_blocks)

    

    

    
    