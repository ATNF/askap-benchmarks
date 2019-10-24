#!/usr/bin/env python
import sys
from pylab import *

# data are dim x dim 32-bit floats
dfile = sys.argv[1]

print 'loading image', dfile
im = np.fromstring(file(dfile, 'rb').read(), np.float32).astype(float)
dim = int(np.sqrt(im.size))
im.shape = dim, dim
print "image shape is", im.shape

imshow(im, cmap=cm.jet)
axis('off')
show()

