import ast
from typing import List
from functools import reduce
import inspect

import numpy as np

from rubick.relation import Relation


class Iterator:
    def __init__(self, id, n):
        self.id = id
        self.n = n
        self.name: str
        self.range: List[int]

    def __add__(self, other):
        return IteratorExpr(1, iterator=self.id, n=self.n) + IteratorExpr(other, n=self.n)

    def __radd__(self, other):
        return IteratorExpr(1, iterator=self.id, n=self.n) + IteratorExpr(other, n=self.n)

    def __sub__(self, other):
        return IteratorExpr(1, iterator=self.id, n=self.n) - IteratorExpr(other, n=self.n)

    def __rsub__(self, other):
        return IteratorExpr(-1, iterator=self.id, n=self.n) + IteratorExpr(other, n=self.n)

    def __mul__(self, other):
        if isinstance(other, int):
            return IteratorExpr(other, iterator=self.id, n=self.n)
        raise TypeError("Unknown multiplier")

    def __rmul__(self, other):
        if isinstance(other, int):
            return IteratorExpr(other, iterator=self.id, n=self.n)
        raise TypeError("Unknown multiplier")

    def setRange(self, lower, upper):
        self.range = [lower, upper]

    def getSize(self):
        return self.range[1] - self.range[0]


class IteratorExpr:
    def __init__(self, val, n, iterator=None, const=0):
        if iterator is not None:
            if isinstance(val, int):
                self.expr = np.zeros(n)
                self.expr[iterator] = val
            else:
                raise TypeError("IteratorExpr: Unknown init")
        else:
            if isinstance(val, IteratorExpr):
                self.expr = val.expr
                self.n = val.n
                self.const = val.const
            elif isinstance(val, Iterator):
                self.expr = np.zeros(n)
                self.expr[val.id] = 1
                self.const = 0
            elif isinstance(val, np.ndarray):
                self.expr = val
            else:
                raise TypeError("IteratorExpr: Unknown init")

        self.n = n
        self.const = const

    def __add__(self, other):
        if isinstance(other, int):
            return IteratorExpr(self.expr, self.n, const=self.const + other)
        if isinstance(other, Iterator):
            return other.__radd__(self)
        if isinstance(other, IteratorExpr):
            if self.n != other.n:
                raise RuntimeError("Incompatible iterator exprs")
            return IteratorExpr(self.expr + other.expr, self.n, const=self.const + other.const)
        raise TypeError("IteratorExpr: Unknown operand")

    def __radd__(self, other):
        if isinstance(other, int):
            return IteratorExpr(self.expr, self.n, const=self.const + other)
        raise TypeError("IteratorExpr: Unknown operand")

    def __sub__(self, other):
        if isinstance(other, int):
            return IteratorExpr(self.expr, self.n, const=self.const - other)
        if isinstance(other, Iterator):
            return other.__rsub__(self)
        if isinstance(other, IteratorExpr):
            if self.n != other.n:
                raise RuntimeError("Incompatible iterator exprs")
            return IteratorExpr(self.expr - other.expr, self.n, const=self.const + other.const)
        raise TypeError("IteratorExpr: Unknown operand")

    def __rsub__(self, other):
        if isinstance(other, int):
            return IteratorExpr(-self.expr, self.n, const=other - self.const)
        raise TypeError("IteratorExpr: Unknown operand")

    def __mul__(self, other):
        if isinstance(other, int):
            return IteratorExpr(other * self.expr, self.n, const=other * self.const)
        if isinstance(other, Iterator):
            return other.__rmul__(self)
        raise TypeError("IteratorExpr: Unknown operand")

    def __rmul__(self, other):
        if isinstance(other, int):
            return IteratorExpr(other * self.expr, self.n, const=other * self.const)
        raise TypeError("IteratorExpr: Unknown operand")


class Tensor:
    def __init__(self, id):
        self.id = id
        self.isOutput = False
        self.accFunc: Relation
        self.name: str

    def __getitem__(self, idx):
        if isinstance(idx, Iterator):
            idx = IteratorExpr(1, n=idx.n, iterator=idx.id)
        if not isinstance(idx, IteratorExpr):
            raise TypeError("Tensor: Unknown index")
        return TensorIndex(self, idx)

    def setRange(self, *args):
        self.range = args

    def domainStr(self):
        indices = [f"i{i+1}" for i in range(len(self.range))]
        return "{" + f"{self.name}[{','.join(indices)}]:{' and '.join([f'0<={i}<{r}' for (i,r) in zip(indices, self.range)])}" + "}"


class TensorExpr:
    def __init__(self):
        self.sons = []

    def __add__(self, other):
        if not isinstance(other, TensorExpr):
            raise TypeError("TensorExpr: Unknown operand")
        return TensorAddExpr(self, other)

    def __mul__(self, other):
        if not isinstance(other, TensorExpr):
            raise TypeError("TensorExpr: Unknown operand")
        return TensorMulExpr(self, other)

    def getAccFunc(self):
        return reduce(dictMerge, map(lambda x: x.getAccFunc(), self.sons))


class TensorIndex(TensorExpr):
    def __init__(self, tensor, idx, prev=None):
        super(TensorIndex, self).__init__()
        self.tensor = tensor
        self.idx = idx
        if prev is None:
            self.mat = [idx.expr.tolist()]
            self.const = [idx.const]
        else:
            if not isinstance(prev, TensorIndex):
                raise TypeError("TensorIndex: Unknown prev")
            self.mat = prev.mat + [idx.expr.tolist()]
            self.const = prev.const + [idx.const]

    def __getitem__(self, idx):
        if isinstance(idx, Iterator):
            idx = IteratorExpr(1, n=idx.n, iterator=idx.id)
        if not isinstance(idx, IteratorExpr):
            raise TypeError("TensorIndex: Unknown index")
        return TensorIndex(self.tensor, idx, self)

    def getAccFunc(self):
        # return {self.tensor.name: {"AccFunc": self.mat, "Const": self.const}}
        return {self.tensor.name: {"accFunc": self.mat}}


class TensorBinaryExpr(TensorExpr):
    def __init__(self, son1, son2):
        super(TensorBinaryExpr, self).__init__()
        self.sons = [son1, son2]


class TensorAddExpr(TensorBinaryExpr):
    def __init__(self, son1, son2):
        super(TensorAddExpr, self).__init__(son1, son2)


class TensorMulExpr(TensorBinaryExpr):
    def __init__(self, son1, son2):
        super(TensorMulExpr, self).__init__(son1, son2)


class OpSpec:
    def __init__(self, name: str):
        self.name = name
        self.numIter = 0
        self.numTensor = 0
        self.iterators: List[Iterator]
        self.tensors: List[Tensor]
        self.output: str

    def genIterators(self, numIter):
        self.numIter = numIter
        self.iterators = [Iterator(i, numIter) for i in range(numIter)]
        context = inspect.stack()[1].code_context[0].strip()
        names = ast.parse(context).body[0].targets[0].elts
        for i in range(len(names)):
            self.iterators[i].name = names[i].id
        self.name2iterator = {
            self.iterators[i].name: i
            for i in range(len(self.iterators))
        }
        return self.iterators

    def getIterator(self, nameOrId):
        if isinstance(nameOrId, int):
            return self.iterators[nameOrId]
        else:
            return self.iterators[self.name2iterator[nameOrId]]

    def genTensors(self, numTensor):
        self.numTensor = numTensor
        self.tensors = [Tensor(i) for i in range(numTensor)]
        context = inspect.stack()[1].code_context[0].strip()
        names = ast.parse(context).body[0].targets[0].elts
        for i in range(len(names)):
            self.tensors[i].name = names[i].id
        self.name2tensor = {
            self.tensors[i].name: i
            for i in range(len(self.tensors))
        }
        return self.tensors

    def getTensor(self, nameOrId):
        if isinstance(nameOrId, int):
            return self.tensors[nameOrId]
        else:
            return self.tensors[self.name2tensor[nameOrId]]

    def makeIndices(self):
        return [i.name for i in self.iterators]

    def setExpr(self, leftValue, rightValue):
        self.expr = rightValue
        self.output = leftValue
        data = dictMerge(self.expr.getAccFunc(), leftValue.getAccFunc())
        leftValue.tensor.isOutput = True

        inIndices = self.makeIndices()

        for u, v in data.items():
            self.getTensor(u).accFunc = Relation(
                "S", inIndices, u, np.array(v['accFunc']))

        self.output = leftValue.tensor.name


def dictMerge(x, y):
    z = {}
    for u, v in x.items():
        z[u] = v
    for u, v in y.items():
        z[u] = v
    return z
