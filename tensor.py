import numpy as np
import sys

class Tensor():
    def __init__(self, dimension, order, basis, components):
        self.dimension = dimension
        self.order = order
        self.basis = self.checkShapeBasis(
                            basis=basis,
                            dimension=dimension,
                            order=order)
        self.components = self.checkShapeComponents(components)

    def in_basis(self, basis):
        basis = self.checkShapeBasis(
                        basis=basis,
                        dimension=self.dimension,
                        order=self.order)

        dualBasis = np.linalg.inv(basis).T
       #  print('dualBasis')
       #  print(dualBasis)
       #  print('self.components')
       #  print(self.components)
       #  #components = np.einsum('ij,k->i', dualBasis, self.components)
        components = [ np.dot(dualBasis[i],self.components) for i in range(self.dimension)]
       #  print('components')
       #  print(components)
        t = Tensor(
                dimension=self.dimension,
                order=self.order,
                basis=basis,
                components=components)
        return t

    def checkShapeBasis(self, basis, dimension, order):
        basis = np.array(basis, dtype = 'double')
        shape = basis.shape
        if len(shape) == 2:
            shape = (1,shape[0],shape[1])
        shapeExpected = (order, dimension,
                dimension)
        self.checkShape(shape, shapeExpected)
        return basis

    def checkShapeComponents(self, comp):
        comp = np.array(comp, dtype = 'double')
        shape = comp.shape
        shapeExpected = tuple((self.dimension for i in range(
                            self.order)))
        self.checkShape(shape, shapeExpected)
        return comp

    def checkShape(self, shape, shapeExpected):
        if (shape != shapeExpected):
            raise ValueError('Shape is:{}. Expected shape:{}'\
                    .format(shape, shapeExpected))
        return None

    def __str__(self):
        l = ['dimension:', self.dimension, 
                'order:', self.order,
                'basis:', self.basis,
                'components:', self.components]
        return '\n'.join(map(lambda x:str(x),l))

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':

    dimension = 3
    basis = [[1,0,0],[0,1,0],[0,0,1]]
    components=[1,1,2]
    c = np.array(components)
    t = Tensor(dimension=dimension,
                order=1,
                basis=basis,
                components=components)


    g_i = np.array([[1,0,0],[1,1,0],[0,2,2]])

    tg = t.in_basis(g_i)

    t2 = Tensor(dimension=dimension,
                order=2,
                basis=[basis,basis],
                components=[components for i in range(3)]) 

    t3 = Tensor(dimension=dimension,
                order=3,
                basis=[basis,basis,basis],
                components=[[components for i in range(3)] 
                    for i in range(3)])

    t4 = Tensor(dimension=dimension,
                order=4,
                basis=[basis,basis,basis,basis],
                components=[[[components for i in range(3)] 
                    for i in range(3)] 
                    for i in range(3)])

