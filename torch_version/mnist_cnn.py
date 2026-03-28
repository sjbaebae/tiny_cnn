import torch
import torch.nn.functional as F
from torch import nn
import time
from functools import partial

def _im2col_col2im_indices_along_dim(
    input_d, kernel_d, dilation_d, padding_d, stride_d, device
):
    """Utility function to implement im2col and col2im"""
    blocks_d = input_d + padding_d * 2 - dilation_d * (kernel_d - 1)

    arange_kw = partial(torch.arange, dtype=torch.int64, device=device)

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = arange_kw(0, blocks_d, stride_d).unsqueeze(0)

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = arange_kw(0, kernel_d * dilation_d, dilation_d).unsqueeze(-1)

    # Broadcast and add kernel starting positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    return blocks_d_indices + kernel_grid

class Conv2D(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, kernel_size: int = 3, stride: int = 1, dilation: int = 1, padding: int = 0, bias: bool = True, activation: 'str' = "relu") -> None:
        super().__init__()

        # define parameters (global)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        # define kernel parameters
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise ValueError(f"Kernel size {kernel_size} not supported")

        # define stride parameters
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise ValueError(f"Stride {stride} not supported")

        # define padding parameters
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            raise ValueError(f"Padding {padding} not supported")

        # define dilation parameters
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        elif isinstance(dilation, tuple):
            self.dilation = dilation
        else:
            raise ValueError(f"Dilation {dilation} not supported")

        # define weights (kernel)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, self.kernel_height, self.kernel_width))

        # define bias if exists (bias applied on each convolution across each out channel)
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

        # define activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        elif activation == "tanh":
            self.activation = F.tanh
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "none":
            self.activation = None
        else:
            raise ValueError(f"Activation function {activation} not supported")

    def forward(self, x: torch.tensor) -> torch.tensor:
        pass

    def _convolve_brute(self, x: torch.tensor) -> torch.tensor:
        # get relevant variables
        batch_size, in_channels, in_height, in_width = x.shape
        out_channels, in_channels, kernel_height, kernel_width = self.weight.shape

        # total input size = (in_height + 2 * self.padding_height)
        # pixels covered by a given kernel is (kernel_size - 1) * dilation + 1
        # stride tells you the gap between each kernel application
        

        out_height = (in_height + 2 * self.padding_height - self.dilation_height * (kernel_height - 1) - 1) // self.stride_height + 1
        out_width = (in_width + 2 * self.padding_width - self.dilation_width * (kernel_width - 1) - 1) // self.stride_width + 1

        # okay technically number of legal positions is total spaces heff = (in_height + 2 * padding_size) - keff ( (kernel_size - 1) * dilation + 1 ) + 1) + 1

        # now we want to 0 index because otherwise for a given stride the possible pos is pos % stride == 1 (since that ensures stride to the left). 0 index, and then we can just do pos % stride == 0

        # So then becomes (heff - keff) is 0 indexed position. We can divide by stride now to get all possible positions that have stride * k to the left (hence valid) -> (heff - keff) // stride.

        # but we know 0 position is valid so + 1. 

        # this results in (in_height + 2 * padding_size - true_kernel_size) // stride + 1

        # now do dumb looping

        output = torch.zeros(batch_size, out_channels, out_height, out_width)

        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        # build accumulator
                        acc = 0
                        for ic in range(in_channels):
                            for kh in range(kernel_height):
                                for kw in range(kernel_width):
                                    # first calculate input coordinates (start h and w). 
                                    # note padding exists to the left of any index. Must provide index corrective factor of -padding to get true index since accessible kernel
                                    ih = oh * self.stride_height - self.padding_height + self.dilation_height * kh
                                    iw = ow * self.stride_width - self.padding_width + self.dilation_width * kw

                                    if ih >= 0 and iw >= 0 and ih < in_height and iw < in_width:
                                        # if in range, now lets do the multiply
                                        acc += x[b, ic, ih, iw] * self.weight[oc, ic, kh, kw]

                        if self.bias is not None:
                            acc += self.bias[oc]

                        if self.activation is not None:
                            acc = self.activation(acc)
                        
                        output[b, oc, oh, ow] = acc

        return output

    def convolve_im2col_slow(self, x: torch.Tensor) -> torch.Tensor:
        # okay similar idea but we convert all these into cols

        # this time we want to preapply padding so that we don't have to do boundary checks
        x_padded = F.pad(x, (self.padding_width, self.padding_width, self.padding_height, self.padding_height))
        
        # get relevant variables
        batch_size, in_channels, in_height, in_width = x_padded.shape
        out_channels, in_channels, kernel_height, kernel_width = self.weight.shape

        # total input size = (in_height + 2 * self.padding_height)
        # pixels covered by a given kernel is (kernel_size - 1) * dilation + 1
        # stride tells you the gap between each kernel application
        
        out_height = (in_height - self.dilation_height * (kernel_height - 1) - 1) // self.stride_height + 1
        out_width = (in_width - self.dilation_width * (kernel_width - 1) - 1) // self.stride_width + 1

        # now we want to create the im2col matrix. 
        cols = []

        # reshape to simplify

        k_effh = self.dilation_height * (kernel_height - 1) + 1
        k_effw = self.dilation_width * (kernel_width - 1) + 1

        for oh in range(out_height):
            for ow in range(out_width):
                oh_end = oh * self.stride_height + k_effh
                ow_end = ow * self.stride_width + k_effw
                row = x_padded[:, :, oh * self.stride_height:oh_end:self.dilation_height, ow * self.stride_width:ow_end:self.dilation_width]
                row = row.reshape(batch_size, -1)
                cols.append(row)

        cols = torch.stack(cols, dim=1)

        kernel = self.weight.reshape(out_channels, -1)

        result = cols @ kernel.T

        if self.bias is not None:
            result += self.bias

        if self.activation is not None:
            result = self.activation(result)

        result = result.reshape(batch_size, out_channels, out_height, out_width)

        return result

    def _check_positive(param, param_name, strict=True):
        cond = all(p > 0 for p in param) if strict else all(p >= 0 for p in param)
        torch._check(
            cond, lambda: f"{param_name} should be greater than zero, but got {param}"
        )

    def _check_size_2(param, param_name):
        # check if either int or len <= 2
        torch._check(len(param) <= 2, lambda: f"{param_name} should be size 2")

    def convolve_im2col_fast(self, x: torch.Tensor) -> torch.Tensor:
        self._check_size_2(self.kernel_size, "kernel_size")
        self._check_size_2(self.dilation, "dilation")
        self._check_size_2(self.padding, "padding")
        self._check_size_2(self.stride, "stride")

        self._check_positive(self.kernel_size, "kernel_size")
        self._check_positive(self.dilation, "dilation")
        self._check_positive(self.padding, "padding", strict=False)
        self._check_positive(self.stride, "stride")

        # get relevant variables
        shape = x.shape
        ndims = len(x.shape)

        torch._check(
            ndim in (3,4) and all(d != 0 for d in shape[-3:]),
            lambda: "Expected 3D or 4D (batch mode) tensor for input with possible 0 batch size "
            f"and non-zero dimensions, but got: {tuple(shape)}",
        )

        # output check
        output_shape = tuple(
            (out + 2 * pad - dil * (ker - 1) - 1) // st + 1
            for out, pad, dil, ker, st in zip(
                shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        )

        torch._check(
            all(c > 0 for c in output_shape),
            lambda: f"Given an input withs shape {shape[-2:]}, "
            f"kernel size {self.kernel_size}, dilation {self.dilation}, "
            f"padding {self.padding}, stride {self.stride}, "
            f"the calculated shape of the sliding blocks is of {output_shape} "
            "but each of the components of output shape must be at least 1"
        )

        # base checks completed.

        # if not batched, batch
        if ndims == 3:
            x = x.unsqueeze(0)

        batch_dim, channel_dim, input_h, input_w = x.shape

        # block indices
        blocks_row_indices = _im2col_col2im_indices_along_dim(
            input_h, self.kernel_size[0], self.dilation[0], self.padding[0], self.stride[0], x.device
        )
        blocks_col_indices = _im2col_col2im_indices_along_dim(
            input_w, self.kernel_size[1], self.dilation[1], self.padding[1], self.stride[1], x.device
        )
        

        # pad input. F.pad takes (pad_left, pad_right, pad_top, pad_bottom) [BAD]
        padded_input = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))

        blocks_row_indices = blocks_row_indices.unsqueeze(-1).unsqueeze(-1)
        output = padded_input[:, :, blocks_row_indices, blocks_col_indices] # (B, C, KH, OH, KW, OW)
        output = output.permute(0, 1, 2, 4, 3, 5) # (B, C, KH, KW, OH, OW)

        # get number of blocks for final reshape
        num_blocks_row = blocks_row_indices.size(1)
        num_blocks_col = blocks_col_indices.size(1)

        output = output.reshape(batch_dim, channel_dim * self.kernel_size[0] * self.kernel_size[1], num_blocks_row * num_blocks_col) # now we get (B, true matrix needed for conv (kernel * in_channels), number of such blocks)
        output = output.permute(0, 2, 1) # (B, num_blocks, true matrix needed for conv (kernel * in_channels))

        # self.weight currently is (out_channels, in_channels, kernel_height, kernel_width)
        # convert to flattened
        weight_flat = self.weight.reshape(self.out_channels, -1)
        return output @ weight_flat.T
        
        
        
        

        

        
        
        

        



    




input_image = torch.randn(1, 1, 28, 28)

conv2d_layer = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1, bias=True, activation="relu")

print("===== SPEED TESTS =====")

end_times = {}

print("1. Brute Force")
start = time.time() 
print(conv2d_layer._convolve_brute(input_image).shape)
end_times["brute"] = time.time() - start
print(f"Time: {end_times['brute']:.4f}")

print("2. Im2Col")
start = time.time()
print(conv2d_layer.convolve_im2col_slow(input_image).shape)
end_times["im2col"] = time.time() - start
print(f"Time: {end_times['im2col']:.4f}")


                        
# LEADERBOARD
print("===== LEADERBOARD =====")
leaderboard = sorted(end_times.items(), key=lambda x: x[1])
for i, (method, time) in enumerate(leaderboard):
    print(f"{i+1}. {method}: {time:.4f} | Speedup factor {(leaderboard[-1][1] / time):.2f}x")
                        
                        

                                    
                                    
            