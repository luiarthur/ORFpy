from ort import ORT
from math import sqrt
from utils import argmax

class ORF:
    def __init__(self,param,numTrees=100,ncores=0):
        """
        Constructor for Online Random Forest. For more info: >>> help(ORT)

        One variable (param) is required to construct an ORF:
        - param          : same as that in ORT. see >>> help(ORT)

        Two parameters are optional: 
        - numTrees       : number of trees in forest (default: 100)
        - ncores         : number of cores to use. (default: 0). Currently NOT IMPLEMENTED, but  if ncores > 0, a parallel implementation will be invoked for speed gains. Preferrably using multiple threads. SO, DON'T USE THIS YET!!! See the update function below.
        
        usage:

        orf = ORF(param,numTrees=20)
        """
        self.param = param
        self.classify = param.has_key('numClasses') > 0
        self.numTrees = numTrees
        self.forest = [ORT(param) for i in xrange(numTrees)]
        self.ncores = ncores

    def update(self,x,y):
        """
        updates the random forest by updating each tree in the forest. As mentioned above, this is currently not implemented. Please replace 'pass' (below) by the appropriate implementation.
        """
        if self.ncores > 1:
            # parallel updates
            pass # FIXME
        else:
            # sequential updates
            for tree in self.forest:
                tree.update(x,y)

    def predict(self,x):
        """
        returns prediction (a scalar) of ORF based on input (x) which is a list of input variables

        usage: 

        x = [1,2,3]
        orf.predict(x)
        """
        preds = [tree.predict(x) for tree in self.forest]
        if self.classify:
            cls_counts = [0] * self.param['numClasses']
            for p in preds:
                cls_counts[p] += 1
            return argmax(cls_counts)
        else:
            return sum(preds) / (len(preds)*1.0)

    def predicts(self,X):
        """
        returns predictions (a list) of ORF based on inputs (X) which is a list of list input variables

        usage: 

        X = [ [1,2,3], [2,3,4], [3,4,5] ]
        orf.predict(X)
        """
        return [self.predict(x) for x in X]

    def predStat(self,x,f):
        """
        returns a statistic aka function (f) of the predictions of the trees in the forest given input x.

        usage:

        def mean(xs):
            return sum(xs) / float(len(xs))

        orf.predStat(x,f) # returns same thing as orf.predict(x). You would replace f by some other function (e.g. sd, quantile, etc.) to get more exotic statistics for predictions.
        """
        return f([tree.predict(x) for tree in self.forest])

    def meanTreeSize(self):
        """
        returns mean tree size of trees in forest. usage:

        orf.meanTreeSize()

        Same idea for next 5 methods (for mean and std. dev.)
        """
        return mean(map(lambda ort: ort.tree.size(), self.forest))

    def meanNumLeaves(self):
        return mean(map(lambda ort: ort.tree.numLeaves(), self.forest))

    def meanMaxDepth(self):
        return mean(map(lambda ort: ort.tree.maxDepth(), self.forest))

    def sdTreeSize(self):
        return sd([ort.tree.size() for ort in self.forest])

    def sdNumLEaves(self):
        return sd([ort.tree.numLeaves() for ort in self.forest])

    def sdMaxDepth(self):
        return sd([ort.tree.maxDepth() for ort in self.forest])
    
    def confusion(self,xs,ys):
        """
        creates a confusion matrix based on list of list of inputs xs, and list of responses (ys). Ideally, xs and ys are out-of-sample data.

        usage:

        orf.confusion(xs,ys)
        """
        n = self.param['numClasses']
        assert n > 1, "Confusion matrices can only be obtained for classification data." 
        preds = self.predicts(xs)
        conf = [[0] * n for i in range(n)]
        for (y,p) in zip(ys,preds):
            conf[y][p] += 1
        return conf
    
    def printConfusion(self,conf):
        """
        simply prints the confusion matrix from the previous confusion method.

        usage:
        conf = orf.confusion(xs,ys)
        orf.printConfusion(conf)
        """
        print    "y\pred\t " + "\t".join(map(str,range(self.param['numClasses'])))
        i = 0
        for row in conf:
            print str(i) + "\t" + "\t".join(map(str,row))
            i += 1

# Other functions:
def mean(xs):
    return sum(xs) / (len(xs)*1.0)

def sd(xs): 
    n = len(xs) *1.0
    mu = sum(xs) / n
    return sqrt( sum(map(lambda x: (x-mu)*(x-mu),xs)) / (n-1) )
