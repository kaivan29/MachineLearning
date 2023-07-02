import numpy as np
import torch
import pandas as pd

print(torch.__version__)

print(torch.cuda.is_available())

# scalar values
scalar_tensor = torch.tensor(3)
print(scalar_tensor)

# matrix or a list
list_tensor = torch.tensor([[1, 2, 3], [4,5,6]])
print(list_tensor)
print(list_tensor.shape)

# numpy array to tensor
numpy_array = np.array([[1, 2, 3], [4,5,6]])
tensor = torch.tensor(numpy_array)
print(tensor)

# tensor to numpy array
np_array_back = tensor.numpy()
print(np_array_back)

# pandas series to tensor
df = pd.DataFrame({'numbers':[1,2,3]})
tensor = torch.tensor(df['numbers'].values)
print(tensor)

# tensor to pandas series
pandas_series_back = pd.Series(tensor.numpy())
print(pandas_series_back)

# difference between uniform and normal distribution
# uniform distribution is a probability distribution where all outcomes are equally likely
# normal distribution is a probability distribution that is symmetric about the mean, showing that data near the mean
# are more frequent in occurrence than data far from the mean

# tensor with random values from uniform distribution
uniform_tensor = torch.rand(2,2)
print(uniform_tensor)

# tensor with random values from normal distribution
normal_tensor = torch.randn(2,2)
print(normal_tensor)

# difference between rand and randn: returns a tensor filled with random numbers from a normal distribution with mean
# 0 and variance 1 (also called the standard normal distribution) rand returns a tensor filled with random numbers
# from a uniform distribution on the interval [0,1)

# zero tensor
zero_tensor = torch.zeros(2,2)
print(zero_tensor)

# one tensor
one_tensor = torch.ones(2,2)
print(one_tensor)

# identity tensor
identity_tensor = torch.eye(2,2)
print(identity_tensor)

# Tensor Types
# torch.FloatTensor: 32-bit floating point
float_tensor = torch.FloatTensor([[1, 2, 3], [4,5,6]])
float_tensor_ = torch.tensor([[1, 2, 3], [4,5,6]], dtype=torch.float)
print(float_tensor)
print(float_tensor_)
# torch.DoubleTensor: 64-bit floating point
# torch.HalfTensor: 16-bit floating point
# torch.ByteTensor: 8-bit unsigned integer
# torch.CharTensor: 8-bit signed integer
# torch.ShortTensor: 16-bit signed integer
# torch.IntTensor: 32-bit signed integer
# torch.LongTensor: 64-bit signed integer

# CUDA
# CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia.
# It allows software developers and software engineers to use a CUDA-enabled graphics processing unit (GPU) for
# general purpose processing – an approach termed GPGPU (General-Purpose computing on Graphics Processing Units).
# The CUDA platform is a software layer that gives direct access to the GPU's virtual instruction set and parallel
# computational elements, for the execution of compute kernels.
# CUDA was developed with several design goals:
# - Provide a small set of extensions to standard programming languages, like C, that enable a straightforward
# implementation of parallel algorithms.
# - Provide a platform-independent model for GPU computing, including an ISA and a memory model.
# - Expose hardware features through the API that are not normally accessible to programmers, reduce the need for
# assembly language and make optimizations portable across different GPU architectures.
# - Provide a platform for research into GPU architectures, especially for investigating the scalability of parallel
# algorithms to many-core architectures.
# - Provide a platform for programming novel computing devices, especially for exploring non-traditional programming
# paradigms, such as the data parallel programming model.
# - Provide a low-level interface to GPU hardware and software, allowing implementers and researchers to easily
# innovate in the GPU space.
# - Provide a platform for programming languages that are more productive than C, such as Python, by providing
# abstractions that map efficiently to low-level hardware mechanisms.
# - Provide a platform for domain-specific languages that map efficiently to low-level hardware mechanisms.
cpu_tensor = torch.rand(2,2)
print(cpu_tensor.device)

if torch.cuda.is_available():
    gpu_tensor = torch.rand(2,2).cuda()
    print(gpu_tensor.device)

# tensor operations
# addition
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
c = a + b
print(c)

# subtraction
c = a - b
print(c)

# multiplication
c = a * b
print(c)

# division
c = a / b
print(c)

# dot product
c = torch.dot(a,b)
print(c)

# matrix multiplication
c = torch.mm(a.reshape(3,1), b.reshape(1,3))
print(c)

# transpose
c = torch.t(c)
print(c)

# indexing
print(c[0,0])
print(c[0,:])
print(c[:,0])

# slicing
print(c[0:2,0:2])

indices = [[1,2],[0,1]]
print(c[indices])

# permutation
# how does permute work?
# permute(1,0) means that the first dimension of the tensor will be the second dimension of the new tensor and the
# second dimension of the tensor will be the first dimension of the new tensor
print(c.permute(1,0))

# masking
mask = c > 5
print(mask)
print(mask.shape)

# masking with a tensor
print(c[mask])
print(c[mask].shape)

# argmin
# what arguments does argmin take?
# dim: the dimension to reduce
# keepdim: whether the output tensor has dim retained or not
# out: the output tensor
print(c.argmin(dim=-1, keepdim=True))
# what does dim=-1 mean? it means that the dimension to reduce is the last dimension
# what does dim=1 mean? it means that the dimension to reduce is the second dimension

# use equal to compare two tensors
print(c.equal(c))

# clip
print(c.clip(min=6,max=12))

# save and load
torch.save(c, 'tensor.pt')
c = torch.load('tensor.pt')
print(c)






