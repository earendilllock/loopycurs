from numpy import *
from pylab import *

def lusolve1(LU,b,r):
  n=shape(LU)[0]
  bcopy=b.copy()
  for l in xrange(r):
        
    for k in xrange(n-1):
      for i in xrange(k+1,n):
        bcopy[i,l] = bcopy[i,l]-bcopy[k,l]*LU[i,k]  

  
    for j in xrange(n-1):
      bcopy[n-1-j,l]=bcopy[n-j-1,l]/LU[n-j-1,n-1-j]
      for m in xrange(n-1-j):
        bcopy[m,l]= bcopy[m,l]-bcopy[n-j-1,l]*LU[m,n-1-j]

  
    bcopy[0,l]=bcopy[0,l]/LU[0,0]
  return bcopy

def lusolve(LU,b,r):
  n=shape(LU)[0]
  bcopy=b.copy()
  for l in xrange(r):
        
    for k in xrange(n-1):
      bcopy[k+1:n,l] = bcopy[k+1:n,l]-bcopy[k,l]*LU[k+1:n,k]
  
    for j in xrange(n-1,0,-1):
      bcopy[j,l] = bcopy[j,l]/LU[j,j]
      bcopy[:j,l] = bcopy[:j,l]-bcopy[j,l]*LU[:j,j]
#  
    bcopy[0,l]=bcopy[0,l]/LU[0,0]
  return bcopy

def ludecomp(b):
  a=b.copy()
  n=shape(a)[0]
  for k in xrange(0,n-1):

    a[k+1:,k]=a[k+1:,k]/a[k,k]
    a[k+1:,k+1:n]=a[k+1:,k+1:n] - dot(a[k+1:,k].reshape(-1,1,order = 'F'),a[k,k+1:n].reshape(1,-1, order = 'F'))
  return a

def lu_decomp(b):
  a=b.copy()
  n=shape(a)[0]
  for k in xrange(0,n-1):
    for i in xrange(k+1,n):
      a[i,k]=a[i,k]/a[k,k]
 
    for j in xrange(k+1,n):
      for i in xrange(k+1,n):
        a[i,j]=a[i,j]-a[i,k]*a[k,j]
  return a
