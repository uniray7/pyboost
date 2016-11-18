import numpy
import pbcvt # your module, also the name of your compiled dynamic library file w/o the extension

a = numpy.array([[1., 2., 3.]])
b = numpy.array([[1.],
                 [2.],
                 [3.]])
print(type(pbcvt.dot(a, b))) # should print [[14.]]
print(pbcvt.dot2(a, b)) # should also print [[14.]]
