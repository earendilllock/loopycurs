import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp

def test_triangle_domain(ctx):
  

  knl = lp.make_kernel(ctx.devices[0], 
  [
  "{[i,j]: 0<=i,j<n and i <= j}",
  ],
  "a[i,j] = 17",
  assumptions="n>=1")

  print knl
  print lp.CompiledKernel(ctx, knl).get_highlighted_code()   
  return knl

plt = cl.get_platforms()
nvidia_plat = plt[0]
ctx = cl.Context(nvidia_plat.get_devices())
knl = test_triangle_domain(ctx)
queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
cknl = lp.CompiledKernel(ctx, knl)

n=128
parameters = {"n":n}
evt, (a) = cknl(queue, **parameters)
print a
