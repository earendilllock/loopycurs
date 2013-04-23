# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ## Тестируем возможности loopy
# 
# $c_{ij} = \sum_{k} a_{ik} b_{kj}$
# 
# - Разобраться с запуском **ядра** (kernel)
# - Выяснить, на чем запускается код

# <codecell>

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp

# <codecell>

def right_u(ctx):
    order = "C"
    dtype = np.float32
    n=128
    r=3
    knl = lp.make_kernel(ctx.devices[0],
            "{[i,j,k,s]: 0<=i<%d and 0<=j<%d and 0<=k<%d and 0<=s<%d}" % (n,n,n,r),
           [
            "f[i,s] = sum((j,k), a[i,j,k]*v[j,s]*w[k,s])",
            ],
            [
            lp.GlobalArg("a", dtype, shape=(n, n, n), order=order),
            lp.GlobalArg("v", dtype, shape=(n, r), order=order),
            lp.GlobalArg("w", dtype, shape=(n, r), order=order),
            lp.GlobalArg("f", dtype, shape=(n, r), order=order),
            ],
            name="pravaya")
    knl = lp.split_iname(knl, "i", 16,outer_tag="g.0", inner_tag="l.0")
    knl = lp.split_iname(knl, "s", 3, outer_tag = "l.1")
    knl = lp.split_iname(knl, "j", 16)
    knl = lp.split_iname(knl, "k", 16) #,16,outer_tag="g.2", inner_tag="l.2")
    seq_knl = knl
    return knl


# <markdowncell>

# # Наблюдения
# ```
# read_and_written_args = ( knl.get_read_variables() & knl.get_written_variables() & set(knl.arg_dict))
# ```  
# Проверяет, нет ли аргументов, которые изменяются внутри kernel
# 
# Ура, мы получили OpenCL код (и видимо даже скомпилировали).  
# Теперь нужно выяснить, как же его запустить

# <codecell>

plt = cl.get_platforms()
nvidia_plat = plt[0]
ctx = cl.Context(nvidia_plat.get_devices())
knl = right_u(ctx)
queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
cknl = lp.CompiledKernel(ctx, knl)

# <codecell>

cknl.print_code()

# <codecell>

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

# <codecell>

#fuu=zeros((n,r))
#for i in xrange (0,n):
#    for s in xrange(0,r):
#       l=0
#       for j in xrange(0,n):
#            h=0
##            for k in xrange(0,n):
#                h=h+a[i,j,k]*w[k,s]
#            l=l+h*v[j,s]
#       fuu[i,s]=l

