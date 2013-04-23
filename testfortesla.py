import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp

def test_nbody(ctx):
    order = "C"
    dtype = np.float32
    n=128
    r=3
    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j,k,s]: 0<=i<%d and 0<=j<%d and 0<=k<%d and 0<=s<%d}" % (n,n,n,r),
           [
            #"ax(j) := sum_float32(k, a[i,j,k] * w[k,s])",
            #"in := sum_float32(j, ax(j)*v[j,s])",
            #"f[i,s] = in",
            "f[i,s] = sum_float32((j,k), a[i,j,k]*v[j,s]*w[k,s])",
            ],
            [
            lp.GlobalArg("a", dtype, shape=(n, n, n), order=order),
            lp.GlobalArg("v", dtype, shape=(n, r), order=order),
            lp.GlobalArg("w", dtype, shape=(n, r), order=order),
            lp.GlobalArg("f", dtype, shape=(n, r), order=order),
            ],
            name="pravaya")

    seq_knl = knl
    return knl

plt = cl.get_platforms()
nvidia_plat = plt[0]
ctx = cl.Context(nvidia_plat.get_devices())
knl = test_nbody(ctx)
queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
cknl = lp.CompiledKernel(ctx, knl)

n = 128
r = 3
#a = np.ones((n, n), dtype = np.float32)
#b = np.ones((n, n), dtype = np.float32)
f = np.zeros((n, r), dtype = np.float32)
a = np.random.randn(n, n, n)
v = np.random.randn(n, r)
w = np.random.randn(n,r)

a = a.astype(np.float32)
v = v.astype(np.float32)
w = w.astype(np.float32)
parameters = {"a" : a, "v": v, "w" : w} 

import time
t1 = time.time()
evt, (f) = cknl(queue, **parameters)
print time.time() - t1




