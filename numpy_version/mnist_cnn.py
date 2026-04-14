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

def _check_size_2(variable, name):
    # check if variable has length of 2 (i.e. padding, stride, etc)
    if len(variable) != 2:
        raise ValueError(f"{name} must have length of 2 but got {len(variable)}")

def _is_positive(variable, name):
    # check if variable is positive
    if not all(x > 0 for x in variable):
        raise ValueError(f"{name} must be positive but got {variable}")
    
def _im_2col_fast(x, kernel_size, stride, padding, dilation, pool = None):

    # x is (batch_size, in_channels, in_height, in_width)

    _check_size_2(kernel_size, "kernel_size")
    _check_size_2(dilation, "dilation")
    _check_size_2(padding, "padding")
    _check_size_2(stride, "stride")

    _is_positive(kernel_size, "kernel_size")
    _is_positive(dilation, "dilation")
    _is_positive(padding, "padding")
    _is_positive(stride, "stride")

    # get shape and ndims
    shape = x.shape
    ndims = len(x.shape)

    # check that ndims is 3 or 4 (image) or batched images (other dims cannot be 0 except for batch)
    if ndims not in [3, 4] and all(s > 0 for s in shape[-3:]):
        raise ValueError(f"x must have 3 or 4 dimensions but got {ndims}")

    # calculate output shape check
    output_shape = tuple(
        (shape[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1,
        (shape[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    )

    # output_shape check
    if not all(s > 0 for s in output_shape):
        raise ValueError(
            f"output_shape must be positive but got {output_shape} for input shape {shape}, "
            f"kernel_size {kernel_size}, "
            f"stride {stride}, "
            f"padding {padding}, "
            f"dilation {dilation}"
        )

    # okay input dims and output_shape okay. Batch if not already batched
    if ndims == 3:
        x = x[None, :, :, :]
    
    # get shape and ndims
    batch_size, in_channels, n_rows, n_cols = x.shape
    kernel_height, kernel_width = kernel_size

    # indices determined for padded input so need to pad x
    padded_x = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode="constant")

    # now get indices
    h_indices = _im2col_col2im_indices_along_dim(n_rows, kernel_height, stride[0], padding[0], dilation[0])
    w_indices = _im2col_col2im_indices_along_dim(n_cols, kernel_width, stride[1], padding[1], dilation[1])

    # now collect indices across dim

    # expand h_indices such that we can broadcast with w_indices
    h_indices = h_indices[:, :, None, None] # shape (kernel_height, n_h_blocks, 1, 1)

    blocks = padded_x[:, :, h_indices, w_indices] # shape (batch_size, in_channels, kernel_height, n_h_blocks, kernel_width, n_w_blocks)
    # before reshape must place kernel_dims next to each other so that those indices are grouped
    blocks.transpose(0, 1, 2, 4, 3, 5) # shape (batch_size, in_channels, kernel_height, kernel_width, n_h_blocks, n_w_blocks)


    # now get num blocks col and width
    n_h_blocks = h_indices.shape[1]
    n_w_blocks = w_indices.shape[1]

    # now reshape based on whether we are doing a pooling operation or not

    if pool is None:
        blocks = blocks.reshape(batch_size, in_channels * kernel_size[0] * kernel_size[1], n_h_blocks * n_w_blocks)
        blocks = blocks.transpose(0, 2, 1) # shape (batch_size, n_h_blocks * n_w_blocks, in_channels * kernel_size[0] * kernel_size[1])

        # now in form of (batch, num_blocks, block). Can now matmul. 
        # output shape needed to know how to reshape back
        return blocks, output_shape
    else:
        # you pool for each kernel, transpose back to kernel mode
        blocks = blocks.transpose(0, 1, 3, 5, 2, 4) # shape (batch_size, in_channels, n_h_blocks, n_w_blocks, kernel_height, kernel_width)
        blocks = blocks.reshape(batch_size, in_channels, n_h_blocks, n_w_blocks, -1)
        if pool == "max":
            blocks = blocks.max(axis=-1) # shape (batch_size, in_channels, n_h_blocks, n_w_blocks)
        elif pool == "mean":
            blocks = blocks.mean(axis=-1) # shape (batch_size, in_channels, n_h_blocks, n_w_blocks)
        else:
            raise ValueError(f"pool must be 'max' or 'mean' but got {pool}")

        # output shape not needed for pooling but can be supplied
        output_shape = (batch_size, in_channels, n_h_blocks, n_w_blocks)
        return blocks, output_shape
        
        
            

        
        


    

    
    

    
    