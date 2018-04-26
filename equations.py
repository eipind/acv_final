import numpy


def alpha_z_eq(b, t, l, v):
    return -(numpy.cross(b, t)/(numpy.dot(l, b)*(numpy.cross(v, t))))


def alpha_eq(b,t,l,v,z):
    return -(numpy.cross(b, t)/(z*numpy.dot(l, b)*(numpy.cross(v, t))))


def z_eq(b,t,l,v,a):
    return -(numpy.cross(b, t) / (a * numpy.dot(l, b) * (numpy.cross(v, t))))

