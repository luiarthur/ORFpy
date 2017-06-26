from math import log

def dataRange(X):
    """
    Accepts a list of lists (X) and returns the "column" ranges. e.g.
    
    X = [[8,7,3], 
         [4,1,9],
         [5,6,2]]
    dataRange(X) # returns: [ [4,8], [1,7], [2,9] ]
    """
    def col(j):
        return map(lambda x: x[j], X)

    k = len(X[0]) # number of columns in X
    return map(lambda j: [ min(col(j)), max(col(j)) ], range(k))
    
def argmax(x):
    """
    returns the index of the element is a list which corresponds to the maximum
    """
    return x.index(max(x))

def log2(x):
    """
    log base 2
    """
    return log(x) / log(2)
