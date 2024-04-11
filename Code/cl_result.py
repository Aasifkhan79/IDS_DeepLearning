import math
import numpy

class cl_result:

    def Silhouette(a, b):
        a = a - numpy.mean(a)
        b = b - numpy.mean(b)
        r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum());
        return r

