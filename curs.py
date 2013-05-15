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
  dtype = np.float32
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
  dtype = np.float32
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
  knl = lp.split_iname(knl, "k", 1)
  knl = lp.split_iname(knl, "i", 32)
  knl = lp.split_iname(knl, "j", 32)
  knl = lp.split_iname(knl, "l", 32)

#  print knl
#  print lp.CompiledKernel(ctx, knl).get_highlighted_code()   
  return knl
def Prav_U(ctx):
  order='C'
  dtype = np.float32
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
  dtype = np.float32
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
  dtype = np.float32
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
  dtype = np.float32
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
  dtype = np.float32
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[j,k,alpha,alpha1]: 0<=alpha,alpha1<r and 0<=j,k<n}",
    
  ],
  [
    "l[alpha,alpha1]=sum((j), v[j,alpha]*v[j,alpha1])*sum((k),w[k,alpha]*w[k,alpha1])",
  ],
  [

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
  dtype = np.float32
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[i,k,alpha,alpha1]: 0<=alpha,alpha1<r and 0<=i,k<n}",
    
  ],
  [
    "l[alpha,alpha1]=sum((i), u[i,alpha]*u[i,alpha1])*sum((k),w[k,alpha]*w[k,alpha1])",
  ],
  [

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
  dtype = np.float32
  knl = lp.make_kernel(ctx.devices[0], 
  [
    
    "{[j,i,alpha,alpha1]: 0<=alpha,alpha1<r and 0<=j,i<n}",
    
  ],
  [
    "l[alpha,alpha1]=sum((i), u[i,alpha]*u[i,alpha1])*sum((j),v[j,alpha]*v[j,alpha1])",
  ],
  [

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
  dtype = np.float32
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



v = np.random.randn(n, r).astype(np.float32)
w = np.random.randn(n, r).astype(np.float32)
u = np.random.randn(n, r).astype(np.float32)
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

#queue = cl.CommandQueue(ctx,properties=cl.command_queue_properties.PROFILING_ENABLE)
prav1=zeros((r,n)).astype(np.float32)
prav=cl.array.to_device(queue,prav1)
left1=zeros((r,r)).astype(np.float32)
left=cl.array.to_device(queue,left1)
v = np.random.randn(n, r).astype(np.float32)
w = np.random.randn(n, r).astype(np.float32)
u = np.random.randn(n, r).astype(np.float32)
u2=cl.array.to_device(queue,u)
v2=cl.array.to_device(queue,v)
w2=cl.array.to_device(queue,w)
for trtrtr in xrange(10):
  parameters={"a":a2,"v":v2,"w":w2,"n":n,"r":r,"f":prav}
  evt =  cknl_r_U(queue, **parameters)[0]
  
  parameters={"v": v2, "w" : w2, "n":n,"r":r,"l":left}
  evt= cknl_l_U(queue,**parameters)[0]
  left2=left.get().copy()
  
  
  parameters={"syst":left,"n":r}
  evt= cknl_decompose(queue,**parameters)[0]
 
  prav2=prav.copy()
  parameters={"LU":left,"bcopy":prav,"n":r,"r":n}
  evt,(f)=cknl_solve(queue,**parameters)
  
  f=f[0].get().transpose().copy()
  u2=cl.array.to_device(queue,f)
##########################----V-----################
  parameters={"a":a2,"u":u2,"w":w2,"n":n,"r":r,"f":prav}
  evt =  cknl_r_V(queue, **parameters)[0]
  
  parameters={"u": u2, "w" : w2, "n":n,"r":r,"l":left}
  evt= cknl_l_V(queue,**parameters)[0]
  left2=left.get().copy()
  
  
  parameters={"syst":left,"n":r}
  evt= cknl_decompose(queue,**parameters)[0]
 
  prav2=prav.copy()
  parameters={"LU":left,"bcopy":prav,"n":r,"r":n}
  evt,(f)=cknl_solve(queue,**parameters)
  
  f=f[0].get().transpose().copy()
  v2=cl.array.to_device(queue,f)  
##########################--------W-----------###########
  parameters={"a":a2,"v":v2,"u":u2,"n":n,"r":r,"f":prav}
  evt =  cknl_r_W(queue, **parameters)[0]
  
  parameters={"v": v2, "u" : u2, "n":n,"r":r,"l":left}
  evt= cknl_l_W(queue,**parameters)[0]
  left2=left.get().copy()
  
  
  parameters={"syst":left,"n":r}
  evt= cknl_decompose(queue,**parameters)[0]
 
  prav2=prav.copy()
  parameters={"LU":left,"bcopy":prav,"n":r,"r":n}
  evt,(f)=cknl_solve(queue,**parameters)
  
  f=f[0].get().transpose().copy()
  w2=cl.array.to_device(queue,f)
######################----Norma-------------###########
  parameters = {"v": v2, "w" : w2, "u": u2, "n":n,"r":r} 
  evt, (f) = cknl_get_tensor(queue, **parameters)
  norm=la.norm(a1-f[0].get())
  print norm
