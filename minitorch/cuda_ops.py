# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to compile a function to run on CUDA device.

    Args:
    ----
        fn: Function to compile
        **kwargs: Additional arguments to pass to numba.cuda.jit

    Returns:
    -------
        Device-compiled version of input function

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable[..., Any], **kwargs: Any) -> FakeCUDAKernel:
    """Decorator to compile a function as a CUDA kernel.

    Args:
    ----
        fn: Function to compile as CUDA kernel
        **kwargs: Additional arguments to pass to numba.cuda.jit

    Returns:
    -------
        Compiled CUDA kernel

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Creates an element-wise binary operation using CUDA.

        Args:
        ----
            fn: Binary function that takes two floats and returns a float

        Returns:
        -------
            A function that applies fn element-wise to two tensors

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Creates a reduction function using CUDA.

        Args:
        ----
            fn: Binary reduction function that takes two floats and returns a float
            start: Initial value for the reduction. Defaults to 0.0

        Returns:
        -------
            A function that performs the reduction operation on a tensor along a dimension

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication of two tensors.

        Performs matrix multiplication between tensor a and tensor b. If both inputs are 2D,
        performs standard matrix multiplication. For higher dimensions, follows broadcasting
        rules treating the last two dimensions as matrices.

        Args:
        ----
            a: First input tensor
            b: Second input tensor. The size of b's second-to-last dimension must match
               the size of a's last dimension.

        Returns:
        -------
            A new tensor containing the matrix product. The output shape is the broadcast
            of a's and b's shapes (excluding the last 2 dimensions) plus [a.shape[-2], b.shape[-1]].

        Raises:
        ------
            AssertionError: If the inner dimensions of the tensors don't match
                (a.shape[-1] != b.shape[-2]).

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i < out_size:
            # Convert the linear index 'i' to multidimensional index 'out_index'
            to_index(i, out_shape, out_index)

            # Handle broadcasting by adjusting 'in_index'
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Calculate the positions in storage
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)

            # Apply the function and store the result
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i < out_size:
            # Convert the linear index 'i' to multidimensional index 'out_index'
            to_index(i, out_shape, out_index)

            # Handle broadcasting for both inputs
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Calculate the positions in storage
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            # Apply the function and store the result
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    # BLOCK_DIM = 32

    # cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    BLOCK_DIM = cuda.blockDim.x  # Number of threads per block
    cache = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)

    # Global and local thread indices
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x + tid

    # Initialize cache with input values or zero if out of bounds
    if i < size:
        cache[tid] = a[i]
    else:
        cache[tid] = 0.0

    # Synchronize threads within the block
    cuda.syncthreads()

    # Perform reduction within the block
    stride = BLOCK_DIM // 2
    while stride > 0:
        if tid < stride:
            cache[tid] += cache[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # Write the result of this block's sum to the output
    if tid == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Performs a parallel sum reduction on the input tensor using CUDA.

    This function divides the input tensor into blocks and performs a sum reduction
    within each block using shared memory. The results of each block's sum are stored
    in the output tensor.

    Args:
    ----
        a (Tensor): The input tensor to be summed.

    Returns:
    -------
        TensorData: A tensor containing the sum of the input tensor elements.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # BLOCK_DIM = 1024
        # cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        # out_index = cuda.local.array(MAX_DIMS, numba.int32)
        # out_pos = cuda.blockIdx.x
        # pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        BLOCK_DIM = cuda.blockDim.x  # Number of threads per block
        cache = cuda.shared.array(THREADS_PER_BLOCK, numba.float64)

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Global thread index across the output tensor
        i = cuda.blockIdx.x

        # Compute the multidimensional index for the output
        to_index(i, out_shape, out_index)

        # Position in the output storage
        out_pos = index_to_position(out_index, out_strides)

        # Initialize the reduction value
        acc = reduce_value

        # Reduction over the specified dimension
        reduce_size = a_shape[reduce_dim]
        tid = cuda.threadIdx.x
        # Each block handles one output element, threads in the block reduce over reduce_dim
        for s in range(tid, reduce_size, BLOCK_DIM):
            # Prepare the input index
            for d in range(len(a_shape)):
                a_index[d] = out_index[d]
            a_index[reduce_dim] = s  # Vary along the reduce dimension

            # Position in the input storage
            a_pos = index_to_position(a_index, a_strides)

            # Apply the reduction function
            acc = fn(acc, a_storage[a_pos])

        # Store partial results in shared memory
        cache[tid] = acc
        cuda.syncthreads()

        # Perform reduction within the block
        stride = BLOCK_DIM // 2
        while stride > 0:
            if tid < stride:
                cache[tid] = fn(cache[tid], cache[tid + stride])
            cuda.syncthreads()
            stride //= 2

        # Write the result to the output
        if tid == 0:
            out[out_pos] = cache[0]

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    # Commented out for now to avoid style issues
    # BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    raise NotImplementedError("Need to implement for Task 3.3")


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform matrix multiplication on two square matrices using CUDA.

    This function multiplies two square matrices `a` and `b` of the same size
    and returns the result as a new tensor. The computation is performed using
    a CUDA kernel.

    Args:
    ----
        a (Tensor): The first input tensor of shape (size, size).
        b (Tensor): The second input tensor of shape (size, size).

    Returns:
    -------
        TensorData: The result of the matrix multiplication as a new tensor.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Commented out for now to avoid style issues
    # a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    # b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # # Batch dimension - fixed
    # batch = cuda.blockIdx.z

    # BLOCK_DIM = 32
    # a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # # The final position c[i, j]
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # # The local position in the block.
    # pi = cuda.threadIdx.x
    # pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    raise NotImplementedError("Need to implement for Task 3.4")


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
