#!/home/earendilllock/tmp/epd-7.3-2-rh5-x86_64/bin/python

import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp


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
    knl = lp.split_iname(knl, "i", 2,outer_tag = "g.0", inner_tag="l.1")

    knl = lp.split_iname(knl, "s", r, outer_tag = "g.1", inner_tag = "l.0")
    knl = lp.split_iname(knl, "j", 16)
    knl = lp.split_iname(knl, "k", 16) #,16,outer_tag="g.2", inner_tag="l.2")
#    knl = lp.add_prefetch(knl, "a", ["k_inner","j_inner", "i_inner"])
    knl = lp.add_prefetch(knl, "v", ["s_inner", "j_inner"])
    knl = lp.add_prefetch(knl, "w", ["s_inner", "k_inner"])
    #knl = lp.add_prefetch(knl, "a", ["k_inner", "j_inner"])
    #knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"])
    seq_knl = knl
    return knl



plt = cl.get_platforms()
nvidia_plat = plt[0]
ctx = cl.Context(nvidia_plat.get_devices())
knl = right_u(ctx)
queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
cknl = lp.CompiledKernel(ctx, knl)
cknl.print_code()



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
knl_gen = lp.generate_loop_schedules(knl)
ref_compiled = lp.CompiledKernel(ctx,knl_gen,options=[],codegen_kwargs={})

a1 = cl_array.to_device(queue,a)
v1 = cl_array.to_device(queue,v)
w1 = cl_array.to_device(queue,w)
f = cl_array.to_device(queue,f)
qq = {"a":a1,"v":v1,"w":w1,'f':f}

from pymbolic import evaluate
kk = {}
for arg in knl.args:
    kk[arg.name] = qq[arg.name].astype(arg.dtype)

import time

t1 = time.time()
for i in xrange(0,100):
    t1 = time.time()
    frfr = ref_compiled(queue,**kk)

    eve=frfr[0]
    eve.wait()

    print time.time() - t1

#import time
#t1 = time.time()
#evt, (f) = cknl(queue, **parameters)

#print time.time() - t1



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

