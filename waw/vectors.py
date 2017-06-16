import numpy as np

#rotate vector
def Rotvec(v, axis, angle):
	I = (axis, (axis+1)%3, (axis+2)%3)
	
	v_r = np.zeros(3)
	
	v_r[I[0]] = v[I[0]]
	v_r[I[1]] = v[I[1]]*np.cos(angle) - v[I[2]]*np.sin(angle)
	v_r[I[2]] = v[I[2]]*np.cos(angle) + v[I[1]]*np.sin(angle)
	
	return v_r

#cross product
def CrossP(v1, v2):
	
	v = np.zeros(3)
	
	for i in range(3):
		v[i] = v1[(i+1)%3]*v2[(i+2)%3] - v1[(i+2)%3]*v2[(i+1)%3]
	
	return(v)

#dot product
def DotP(v1,v2):
	
	return(np.sum(v1*v2))

#area of a triangle embedded in 3D space
def TriArea(A,B,C):
	
	"""Returns the area of the triangle whose vertices are the given by the three dimensional vectors A,B and C"""
	
	#define the side vectors
	a = B-A
	b = C-A
	
	#compute cross product
	n = CrossP(a,b)
	
	#area is half the modulus of the cross product
	return(0.5*np.sqrt(np.sum(n*n)))

#area of a triangle on a plane
def TriArea2D(A,B,C):
	
	"""Returns the area of the triangle whose vertices are the given by the two dimensional vectors A,B and C"""
	
	#define the side vectors
	a = B-A
	b = C-A
	
	return(0.5*abs(a[0]*b[1] - a[1]*b[0]))

#affine linear extension: interpolation of f in x (point inside a 
#triangle) based on f(A),f(B) and f(C) (values at the vertices)
def AffineLinearExtension(x,A,B,C,fA,fB,fC):
	
	"""Computes the interpolated value f(x) based on the values of f in A,B,C, where x is a point inside a triangle and A,B,C are the vertices of that triangle."""
	
	Area = TriArea(A,B,C)
	
	if Area>0.:
		return((TriArea(B,C,x)*fA + TriArea(C,A,x)*fB + TriArea(A,B,x)*fC)/TriArea(A,B,C))
	else:
		return(fA)

