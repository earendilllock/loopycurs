from numpy import *
from pylab import *
from test import *
import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array as cl_array
import loopy as lp

def LU_decomposition(ctx):
  order='C'
  dtype = np.float64
  knl = lp.make_kernel(ctx.devices[0], 
  [
    "{[k,i]: 0<=k<n-1 and k+1<=i<n}",
    "{[j,l]: 0<=k<n-1 and k+1<=j,l<n}",
  ],
  [
  "syst[i,k] = syst[i,k]/syst[k,k] {id=lab1}",
  "syst[l,j]= syst[l,j] - syst[l,k]*syst[k,j] {dep=lab1}",
  ],
  [
  lp.GlobalArg("syst", dtype, shape = "n, n" , order=order),
  lp.ValueArg("n", np.int32),
  ],
  assumptions="n>=1")
  knl = lp.split_iname(knl, "k", n)
  knl = lp.split_iname(knl, "i", 32)
  knl = lp.split_iname(knl, "j", 32)
  knl = lp.split_iname(knl, "l", 32)

#  print knl
#  print lp.CompiledKernel(ctx, knl).get_highlighted_code()   
  return knl

def LU_solver(ctx):
  order='C'
  dtype = np.float64
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[l,k,i,j,m]: 0<=l<r and 0<=k<n-1 and k+1<=i<n and 0<=j<n-1 and 0<=m<n-1-j}",
    
  ],
  [
  "bcopy[i,l] = bcopy[i,l]-bcopy[k,l]*LU[i,k] {id=lab1}",
  "bcopy[n-1-j,l]=bcopy[n-j-1,l]/LU[n-j-1,n-1-j] {id=l2, dep=lab1}",
  "bcopy[m,l]= bcopy[m,l]-bcopy[n-j-1,l]*LU[m,n-1-j] {id=l3, dep =l2}",
  "bcopy[0,l]=bcopy[0,l]/LU[0,0]{id=l4, dep=l2}",
  ],
  [
  lp.GlobalArg("LU", dtype, shape = "n, n" , order=order),
  lp.GlobalArg("bcopy", dtype, shape = "n, r" , order=order),
  lp.ValueArg("n", np.int64),
  lp.ValueArg("r", np.int64),
  ],
  assumptions="n>=1")
  knl = lp.split_iname(knl, "k", n)
  knl = lp.split_iname(knl, "i", 32)
  knl = lp.split_iname(knl, "j", 32)
  knl = lp.split_iname(knl, "l", 32)

#  print knl
#  print lp.CompiledKernel(ctx, knl).get_highlighted_code()   
  return knl
def Prav_U(ctx):
  order='C'
  dtype = np.float64
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[i,j,k,alpha]: 0<=alpha<r and 0<=i,j,k<n}",
    
  ],
  [
    "f[alpha,i]=sum((j,k), a[i,j,k]*v[j,alpha]*w[k,alpha])",
  ],
  [
    lp.GlobalArg("a", dtype, shape="n, n, n", order=order),
    lp.GlobalArg("v", dtype, shape="n, r", order=order),
    lp.GlobalArg("w", dtype, shape="n, r", order=order),
    lp.GlobalArg("f", dtype, shape="r, n", order=order),
    lp.ValueArg("n", np.int64),
    lp.ValueArg("r", np.int64),
  ],
  assumptions="n>=1")
  knl = lp.split_iname(knl, "i", 16,outer_tag="g.0", inner_tag="l.0")
  knl = lp.split_iname(knl, "alpha", 3, outer_tag="g.1", inner_tag="l.1")
  knl = lp.split_iname(knl, "j", 16)
  knl = lp.split_iname(knl, "k", 16)
  
  return knl

def Prav_V(ctx):
  order='C'
  dtype = np.float64
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[i,j,k,alpha]: 0<=alpha<r and 0<=i,j,k<n}",
    
  ],
  [
    "f[alpha,j]=sum((k,i), a[i,j,k]*w[k,alpha]*u[i,alpha])",
  ],
  [
    lp.GlobalArg("a", dtype, shape="n, n, n", order=order),
    lp.GlobalArg("u", dtype, shape="n, r", order=order),
    lp.GlobalArg("w", dtype, shape="n, r", order=order),
    lp.GlobalArg("f", dtype, shape="r, n", order=order),
    lp.ValueArg("n", np.int64),
    lp.ValueArg("r", np.int64),
  ],
  assumptions="n>=1")
  knl = lp.split_iname(knl, "j", 16,outer_tag="g.0", inner_tag="l.0")
  knl = lp.split_iname(knl, "alpha", 3, outer_tag="g.1", inner_tag="l.1")
  knl = lp.split_iname(knl, "i", 16)
  knl = lp.split_iname(knl, "k", 16) 
   
  return knl

def Prav_V(ctx):
  order='C'
  dtype = np.float64
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[i,j,k,alpha]: 0<=alpha<r and 0<=i,j,k<n}",
    
  ],
  [
    "f[alpha,j]=sum((k,i), a[i,j,k]*w[k,alpha]*u[i,alpha])",
  ],
  [
    lp.GlobalArg("a", dtype, shape="n, n, n", order=order),
    lp.GlobalArg("u", dtype, shape="n, r", order=order),
    lp.GlobalArg("w", dtype, shape="n, r", order=order),
    lp.GlobalArg("f", dtype, shape="r, n", order=order),
    lp.ValueArg("n", np.int64),
    lp.ValueArg("r", np.int64),
  ],
  assumptions="n>=1")
  knl = lp.split_iname(knl, "j", 16,outer_tag="g.0", inner_tag="l.0")
  knl = lp.split_iname(knl, "alpha", 3, outer_tag="g.1", inner_tag="l.1")
  knl = lp.split_iname(knl, "i", 16)
  knl = lp.split_iname(knl, "k", 16) 
   
  return knl

def Prav_W(ctx):
  order='C'
  dtype = np.float64
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[i,j,k,alpha]: 0<=alpha<r and 0<=i,j,k<n}",
    
  ],
  [
    "f[alpha,k]=sum((i,j), a[i,j,k]*u[i,alpha]*v[j,alpha])",
  ],
  [
    lp.GlobalArg("a", dtype, shape="n, n, n", order=order),
    lp.GlobalArg("v", dtype, shape="n, r", order=order),
    lp.GlobalArg("u", dtype, shape="n, r", order=order),
    lp.GlobalArg("f", dtype, shape="r, n", order=order),
    lp.ValueArg("n", np.int64),
    lp.ValueArg("r", np.int64),
  ],
  assumptions="n>=1")
  knl = lp.split_iname(knl, "k", 16,outer_tag="g.0", inner_tag="l.0")
  knl = lp.split_iname(knl, "alpha", 3, outer_tag="g.1", inner_tag="l.1")
  knl = lp.split_iname(knl, "j", 16)
  knl = lp.split_iname(knl, "i", 16) 
  

  return knl

def left_U(ctx):
  order='C'
  dtype = np.float64
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[j,k,alpha,alpha1]: 0<=alpha,alpha1<r and 0<=j,k<n}",
    
  ],
  [
    "l[alpha,alpha1]=sum((j), v[j,alpha]*v[j,alpha1])*sum((k),w[k,alpha]*w[k,alpha1])",
  ],
  [
    lp.GlobalArg("a", dtype, shape="n, n, n", order=order),
    lp.GlobalArg("v", dtype, shape="n, r", order=order),
    lp.GlobalArg("w", dtype, shape="n, r", order=order),
    lp.GlobalArg("l", dtype, shape="r, r", order=order),
    lp.ValueArg("n", np.int64),
    lp.ValueArg("r", np.int64),
  ],
  assumptions="n>=1")
  knl = lp.split_iname(knl, "alpha1", 16,outer_tag="g.0", inner_tag="l.0")
  knl = lp.split_iname(knl, "alpha", 3, outer_tag="g.1", inner_tag="l.1")
  knl = lp.split_iname(knl, "j", 16)
  knl = lp.split_iname(knl, "k", 16)
  
  return knl

def left_V(ctx):
  order='C'
  dtype = np.float64
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[i,k,alpha,alpha1]: 0<=alpha,alpha1<r and 0<=i,k<n}",
    
  ],
  [
    "l[alpha,alpha1]=sum((i), u[i,alpha]*u[i,alpha1])*sum((k),w[k,alpha]*w[k,alpha1])",
  ],
  [
    lp.GlobalArg("a", dtype, shape="n, n, n", order=order),
    lp.GlobalArg("u", dtype, shape="n, r", order=order),
    lp.GlobalArg("w", dtype, shape="n, r", order=order),
    lp.GlobalArg("l", dtype, shape="r, r", order=order),
    lp.ValueArg("n", np.int64),
    lp.ValueArg("r", np.int64),
  ],
  assumptions="n>=1")
  knl = lp.split_iname(knl, "alpha1", 16,outer_tag="g.0", inner_tag="l.0")
  knl = lp.split_iname(knl, "alpha", 3, outer_tag="g.1", inner_tag="l.1")
  knl = lp.split_iname(knl, "i", 16)
  knl = lp.split_iname(knl, "k", 16)
  
  return knl

def left_W(ctx):
  order='C'
  dtype = np.float64
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[j,i,alpha,alpha1]: 0<=alpha,alpha1<r and 0<=j,i<n}",
    
  ],
  [
    "l[alpha,alpha1]=sum((i), u[i,alpha]*u[i,alpha1])*sum((j),v[j,alpha]*v[j,alpha1])",
  ],
  [
    lp.GlobalArg("a", dtype, shape="n, n, n", order=order),
    lp.GlobalArg("v", dtype, shape="n, r", order=order),
    lp.GlobalArg("u", dtype, shape="n, r", order=order),
    lp.GlobalArg("l", dtype, shape="r, r", order=order),
    lp.ValueArg("n", np.int64),
    lp.ValueArg("r", np.int64),
  ],
  assumptions="n>=1")
  knl = lp.split_iname(knl, "alpha1", 16,outer_tag="g.0", inner_tag="l.0")
  knl = lp.split_iname(knl, "alpha", 3, outer_tag="g.1", inner_tag="l.1")
  knl = lp.split_iname(knl, "j", 16)
  knl = lp.split_iname(knl, "i", 16)
  
  return knl

def get_tensor(ctx):
  order='C'
  dtype = np.float64
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[j,i,alpha,k]: 0<=alpha<r and 0<=i,j,k<n}",
    
  ],
  [
    "res[i,j,k]=sum((alpha), u[i,alpha]*v[j,alpha]*w[k,alpha])",
  ],
  [
    lp.GlobalArg("res", dtype, shape="n, n, n", order=order),
    lp.GlobalArg("v", dtype, shape="n, r", order=order),
    lp.GlobalArg("u", dtype, shape="n, r", order=order),
    lp.GlobalArg("w", dtype, shape="n, r", order=order),
    lp.ValueArg("n", np.int32),
    lp.ValueArg("r", np.int32),
  ],
  assumptions="n>=1")
  knl = lp.split_iname(knl, "i", 8,outer_tag="g.0", inner_tag="l.0")
  knl = lp.split_iname(knl, "j", 8, outer_tag="g.1", inner_tag="l.1")
  knl = lp.split_iname(knl, "alpha", 2)
  knl = lp.split_iname(knl, "k", 8, outer_tag="g.2", inner_tag="l.2" )
  
  return knl

def solve_it(n,r,ctx,a,b):
      bcopy=b.copy()
     
      decompose_knl = LU_decomposition(ctx)
      queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
      cknl = lp.CompiledKernel(ctx, decompose_knl)
      parameters = {"syst": a, "n": r}
      evt, (LU) = cknl(queue, **parameters)
      
      LU=LU[0].astype(np.float64)
      solve_knl= LU_solver(ctx)
      queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
      cknl = lp.CompiledKernel(ctx, solve_knl)
      parameters = {"LU": LU, "bcopy": bcopy,"n":r, "r":n}
      evt, (c) = cknl(queue, **parameters)
      return c[0].get().transpose().astype(np.float64).copy()

n=128
r=3
k=0
norm2=1
norm=1
eps=1e-8
d=3
dimension=[n,n,n]


plt = cl.get_platforms()
nvidia_plat = plt[0]
ctx = cl.Context(nvidia_plat.get_devices())


knl_get_tensor = get_tensor(ctx)
knl_r_U = Prav_U(ctx)
knl_r_V = Prav_V(ctx)
knl_r_W = Prav_W(ctx)
knl_l_U = left_U(ctx)
knl_l_V = left_V(ctx)
knl_l_W = left_W(ctx)
cknl_r_U = lp.CompiledKernel(ctx, knl_r_U)
cknl_r_V = lp.CompiledKernel(ctx, knl_r_V)
cknl_r_W = lp.CompiledKernel(ctx, knl_r_W)
cknl_l_U = lp.CompiledKernel(ctx, knl_l_U)
cknl_l_V = lp.CompiledKernel(ctx, knl_l_V)
cknl_l_W = lp.CompiledKernel(ctx, knl_l_W)
cknl_get_tensor=lp.CompiledKernel(ctx, knl_get_tensor)
solve_knl= LU_solver(ctx)
decompose_knl = LU_decomposition(ctx)

cknl_decompose = lp.CompiledKernel(ctx, decompose_knl)
cknl_solve = lp.CompiledKernel(ctx, solve_knl)
queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)



v = np.random.randn(n, r).astype(np.float64)
w = np.random.randn(n, r).astype(np.float64)
u = np.random.randn(n, r).astype(np.float64)
ulist=list(arange(d))
ulist[0]=u.copy()
ulist[1]=v.copy()
ulist[2]=w.copy()
u2=cl.array.to_device(queue,u)
v2=cl.array.to_device(queue,v)
w2=cl.array.to_device(queue,w)
parameters = {"v": v2, "w" : w2, "u": u2, "n":n,"r":r} 
evt, (f) = cknl_get_tensor(queue, **parameters)
a1 = f[0].get()
a = f[0]
a2=cl.array.to_device(queue,a1)
v = np.random.randn(n, r).astype(np.float64).copy()
w = np.random.randn(n, r).astype(np.float64).copy()
u = np.random.randn(n, r).astype(np.float64).copy()
queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)

u2=cl.array.to_device(queue,u.copy())
v2=cl.array.to_device(queue,v.copy())
w2=cl.array.to_device(queue,w.copy())

for trtrtr in xrange(10):
    k=k+1
    t=time.time()
    #################-------U-part---------################
    
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"a" : a2, "v": v2, "w" : w2, "n":n,"r":r} 
    evt, (prav) = cknl_r_U(queue, **parameters)
    evt.wait()
    prav=prav[0]
    ch_r=rights(a1,ulist,dimension,d,r,0) 
    print la.norm(prav.get()-ch_r.transpose())
 
    
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"a" : a2, "v": v2, "w" : w2, "n":n,"r":r} 
    evt, (left) = cknl_l_U(queue, **parameters)
    evt.wait()
    left=left[0].get()
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    ch_l=lefts(ulist,0,d,r)
    print la.norm(left-ch_l)
    parameters = {"syst": left, "n": r}
    evt, (LU) = cknl_decompose(queue, **parameters)
    evt.wait()
    #LU=LU[0].get().astype(np.float64)
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"LU": LU[0], "bcopy": prav.get(),"n":r, "r":n}
    evt, (c) = cknl_solve(queue, **parameters)
    evt.wait()
    unew=solve(ch_l,ch_r.transpose()).transpose()
    
    ulist[0]=unew.copy()
    u2=cl.array.to_device(queue,c[0].transpose().copy())
    print "u",la.norm(ulist[0]-u2.get())
    ###################-------V-part----------##############
    
    
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"a" : a2, "u": u2, "w" : w2, "n":n,"r":r} 
    evt, (prav) = cknl_r_V(queue, **parameters)
    evt.wait()
    prav=prav[0]
    ch_r=rights(a1,ulist,dimension,d,r,1) 
    print la.norm(prav.get()-ch_r.transpose())
    #    
    #
    #
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"a" : a2, "u": u2, "w" : w2, "n":n,"r":r} 
    evt, (left) = cknl_l_V(queue, **parameters)
    evt.wait()
    left=left[0].get()
    ch_l=lefts(ulist,1,d,r)
    print la.norm(left-ch_l)
        
    #v=solve_it(n,r,ctx,left,prav)
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"syst": left, "n": r}
    evt, (LU) = cknl_decompose(queue, **parameters)
    evt.wait()
    #LU=LU[0].get().astype(np.float64)
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"LU": LU[0], "bcopy": prav.get(),"n":r, "r":n}
    evt, (c) = cknl_solve(queue, **parameters)
    evt.wait()
    v2=cl.array.to_device(queue,c[0].transpose().copy())
    unew=solve(ch_l,ch_r.transpose()).transpose()
    
    ulist[1]=unew.copy()
    print "v",la.norm(ulist[1]-v2.get())
    #
    #####################------ W-part-------##############
    #
    t2=time.time()
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"a" : a2, "v": v2, "u" : u2, "n":n,"r":r} 

    evt, (prav) = cknl_r_W(queue, **parameters)
    evt.wait()
    prav=prav[0]
    ch_r=rights(a1,ulist,dimension,d,r,2) 
    print la.norm(prav.get()-ch_r.transpose())     
    
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"a" : a2, "v": v2, "u" : u2, "n":n,"r":r} 
    evt, (left) = cknl_l_W(queue, **parameters)
    evt.wait()
    left=left[0].get()

    ch_l=lefts(ulist,2,d,r)
    print la.norm(left-ch_l)        
#    w=solve_it(n,r,ctx,left,prav)
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"syst": left, "n": r}
    evt, (LU) = cknl_decompose(queue, **parameters)
    evt.wait()
    #LU=LU[0].get().astype(np.float64)
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"LU": LU[0], "bcopy": prav.get(),"n":r, "r":n}
    evt, (c) = cknl_solve(queue, **parameters)
    evt.wait()
    w2=cl.array.to_device(queue,c[0].transpose().copy())
    
    unew=solve(ch_l,ch_r.transpose()).transpose()
    
    ulist[2]=unew.copy()
    print "w",la.norm(ulist[2]-w2.get())
    #
############################
    queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
    parameters = {"v": v2, "w" : w2, "u": u2, "n":n,"r":r} 
    evt, (aaa) = cknl_get_tensor(queue, **parameters)
    evt.wait()
    tmp=aaa[0].get()
    a2=cl.array.to_device(queue,tmp)
 
    t=time.time()-t
    print la.norm(a2.get()-a1).astype(float64)

