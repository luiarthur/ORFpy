class Tree:
    """
    # Tree - Binary tree class

    Example:
        from tree import Tree

        t1 = Tree(1)
        t1.draw()
        t2 = Tree(1,Tree(2),Tree(3))
        t3 = Tree(1,t2,Tree(4))
        t4 = Tree(1,t2,t3)
        t4.draw()

        t4.maxDepth()  # should be 4
        t4.size()      # should be 9
        t4.numLeaves() # should be 5

    """
    def __init__(self, elem, left=None, right=None):
        """
        Example:
        t1 = Tree(1)                # creates a tree with a root node (1) and no children
        t2 = Tree(1,Tree(2),Tree(3) # creates a tree with a root node (1) with left tree (2) and right tree(3)
        t3 = Tree(1,t2,Tree(4))     # creates a tree with a root node (1) with left subtree (t2) and right tree(4)

        =====================================================
        Note: Nodes in tree must have exactly 2 or 0 children
        =====================================================
        """
        assert((left == None and right == None) or (left != None and right != None))
        self.elem = elem
        self.left = left
        self.right = right

    def updateChildren(self,l,r):
        """
        updates the left and right children trees.  e.g.

        >>> t = Tree(1)
        >>> t.draw()

        Leaf(1)

        >>> t.updateChildren(Tree(2),Tree(3))
        
        __1__
        2   3
        """
        self.left, self.right = l,r

    def isLeaf(self):
        """
        returns a boolean. True if the Tree has no children, False otherwise
        """
        return self.left == None and self.right == None

    def size(self):
        """
        returns the number of internal nodes in tree
        """
        return 1 if self.isLeaf() else self.left.size() + self.right.size() + 1
    
    def numLeaves(self):
        """
        returns number of leaves in tree
        """
        return  1 if self.isLeaf() else self.left.numLeaves() + self.right.numLeaves()

    def maxDepth(self):
        """
        returns maximum depth of tree
        """
        return self.__md(1)

    def __md(self,s):
        return s if self.isLeaf() else max(self.left.__md(s+1),self.right.__md(s+1))

    def inOrder(self):
        """
        Returns the in-order sequence of tree. Needs to be implemented...
        """
        print "return in-order sequence of tree. needs to be implemented" # FIXME
        pass

    def preOrder(self):
        """
        Returns the pre-order sequence of tree. Needs to be implemented...
        """
        print "return pre-order sequence of tree. needs to be implemented" # FIXME
        pass

    def draw(self):
        """
        Draw the tree in a pretty way in the console. Good for smaller trees. You probably don't want to draw a very large tree...
        """
        print self.treeString()

    def treeString(self,fun=False):
        """
        Returns a string representing the flattened tree
        """
        if fun:
            return "Leaf(" + self.elem.toString() + ")" if self.isLeaf() else "\n" + "\n".join( self.__pretty(spacing=1,fun=fun) ) + "\n"
        else:
            return "Leaf(" + str(self.elem) + ")" if self.isLeaf() else "\n" + "\n".join( self.__pretty(spacing=1,fun=fun) ) + "\n"

    def __pretty(self,spacing=3,fun=False):
        def paste(l, r): # l,r: string lists
            def elongate(ls):
                maxCol = max(map(len,ls))
                return map(lambda  s: s + " "*(maxCol - len(s)) , ls)

            maxRow = max( map(len, [l,r]) )
            tmp = map(lambda x: x + [""]*(maxRow-len(x)), [l,r])
            newL,newR = map(elongate,tmp)
            return [newL[i] + newR[i] for i in xrange(maxRow)]

        ps = self.elem.toString() if fun else str(self.elem)
        ls,rs = map(lambda x: [x.elem.toString() if fun else str(x.elem)] if x.isLeaf() else x.__pretty(spacing,fun), (self.left,self.right))
        posL = ls[0].index(self.left.elem.toString() if fun else str(self.left.elem))
        posR = rs[0].index(self.right.elem.toString() if fun else str(self.right.elem))
        top = " "*posL + "_"*(spacing+len(ls[0])-posL) + ps + "_"*(spacing+posR) + " "*(len(rs[0])-posR)
        bottom = paste(ls, paste([" "*(spacing+len(ps))],rs)) # use reduce?
        return [top] + bottom
