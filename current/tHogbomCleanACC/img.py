#!/usr/bin/env python
import sys
from pylab import *

# data are 4096x4096 32-bit floats
dfile = sys.argv[1]

print 'loading image', dfile
im = np.fromstring(file(dfile, 'rb').read(), np.float32).astype(float)
dim = sqrt(im.size)
im.shape = dim, dim

imshow(im, cmap=cm.jet)
axis('off')
show()

