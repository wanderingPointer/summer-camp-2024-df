import copy
import json
from time import time
from typing import List, Optional, Union, Iterator, Tuple
from functools import reduce
import subprocess
import os

import numpy as np
from pandas import array
from sympy import O

from rubick.relation import Relation, VarSet
from rubick.interface import OpSpec
from rubick.util import upDiv

MAXBUFFERSIZE = 1 << 32


class AccessEntry:
    def __init__(self, arraySpec, attrib):
        self.arraySpec = arraySpec
        self.name = attrib["name"]
        self.relation = Relation.read(attrib["relation"])
        self.mat = self.relation.mat

        self.vecs = [v if isinstance(v, np.ndarray) else np.array(v)
                     for v in attrib["vecs"]]

        self.hasDiag = attrib["hasDiag"]
        self.canInput = attrib["input"]
        self.canOutput = attrib["output"]

        self._checkDimension()

        self._checkReuse()

    def _checkDimension(self):
        pass

    def _checkReuse(self):
        self.systolic = set()
        self.multicast = set()
        self.stationary = False
        for v in self.vecs:
            if v[self.arraySpec.spaceDims] != 0:
                if np.sum(v != 0) > 1:
                    self.systolic.add(v.tolist)
                else:
                    self.stationary = True
            else:
                self.multicast.add(v.tolist)

    # def _memTRReduce(self, reduced):
    #     for i in range(self.mat.shape[0]):
    #         r = [False] * self.spaceDims()
    #         for j in range(0, self.spaceDims()):
    #             if self.mat[i][j] < -1e-5:
    #                 r[j] = True
    #         for j in range(self.spaceDims(), self.mat.shape[1]):
    #             if self.mat[i][j] > 1e-5:
    #                 for k in range(self.spaceDims()):
    #                     reduced[j][k] = r[k] or reduced[j][k]

    # def memoryTimeRangeReduce(self, spaceRange, timeRanges):
    #     reduced = [[False] * self.spaceDims for i in range(len(timeRanges))]
    #     self._memTRReduce(reduced)
    #     result = copy.copy(timeRanges)
    #     for i in range(len(timeRanges)):
    #         for j in range(self.spaceDims()):
    #             if reduced[i][j]:
    #                 result[i - self.spaceDims()] -= spaceRange - 1
    #     return result

    def spaceDims(self):
        return self.arraySpec.spaceDims

    def extendRelation(self, timeDims: int):
        # Extend the dimension of times to timeDims by appending a np.eye(timeDims-1)
        if timeDims < 1:
            raise RuntimeError("AccessEntry: can not extend to timeDims < 1")
        if timeDims == 1:
            return self.relation

        mat = self.relation.mat
        mat = np.vstack([
            np.hstack([mat, np.zeros((mat.shape[0], timeDims - 1))]),
            np.hstack(
                [np.zeros((timeDims - 1, mat.shape[1])), np.eye(timeDims - 1)])
        ])
        indices = self.relation.inIndices + \
            [f"t{i+2}" for i in range(timeDims - 1)]
        return Relation(self.relation.inDom, indices, self.relation.outDom, mat)

    def __str__(self) -> str:
        return self.name


class DecomposeError(RuntimeError):
    def __init__(self, what):
        super(DecomposeError, self).__init__(what)


class ArraySpec:
    def __init__(self, fileName):
        with open(fileName, "r") as fin:
            info = json.load(fin)
            self.spaceDims : int = info["spaceDims"]
            self.spaceNames : List[str] = info["spaceNames"]
            self.spaceRange : List[int] = info["spaceRange"]
            self.dirVecs : List[List[int]] = info["dirVecs"]
            self.entries = [AccessEntry(self, i)
                            for i in info["entries"]]

        self.name2Entry = {i.name: i for i in self.entries}
        self._makeDecomposer()

    def _makeDecomposer(self):
        def findVec(vec):
            for i, v in enumerate(self.dirVecs):
                if np.all(vec == v):
                    return i
            return -1

        for i, v in enumerate(self.dirVecs):
            self.dirVecs[i] = np.array(v)

        self.vecSets = {}
        for e in self.entries:
            self.vecSets[e.name] = set()
            for row in e.vecs:
                v = findVec(row)
                self.vecSets[e.name].add(v)
                if v == -1:
                    raise DecomposeError("Undefined dir-vec " + str(row) +
                                         " in entry " + e.name)

    def lookUpEntry(self, movement: np.ndarray):
        r = self.spaceDims + 1 - np.linalg.matrix_rank(movement)
        found = set()
        for i, v in enumerate(self.dirVecs):
            if np.allclose(movement @ v, np.zeros(movement.shape[0])):
                found.add(i)
            if len(found) == r:
                break

        notEnough = False
        for e in self.entries:
            if self.vecSets[e.name] == found:
                if len(self.vecSets[e.name]) == r:
                    return e
                else:
                    notEnough = True
        if notEnough:
            raise DecomposeError(
                "Potential undefined dir-vec in movement " + str(movement))
        else:
            raise DecomposeError(
                "No matching data entry for movemnet " + str(movement))

    def getEntry(self, name):
        return self.name2Entry[name]


class DataLayout:
    def __init__(self, arraySpec, relation):
        if isinstance(relation, str):
            relation = Relation.read(relation)
        self.relation = relation
        self.mat = relation.mat
        self.arraySpec = arraySpec

        self.inDims = {
            "space": arraySpec.spaceDims,
            "time": self.mat.shape[1] - arraySpec.spaceDims,
        }
        self.outDims = self.mat.shape[0]

    # def bufferSize(self, spaceRange, timeRanges, tensorRanges):
    #     if len(timeRanges) != self.inDims["time"]:
    #         raise RuntimeError(
    #             "Time range length inconsistent with data layout time dimensions")

    #     # ranges = np.concatenate(
    #         [[spaceRange] * self.inDims["space"], timeRanges]) - 1
    #     bound = self.mat @ ranges + 1
    #     result = np.prod(np.fmin(bound, tensorRanges))
    #     return result

    def __str__(self) -> str:
        return str(self.relation)

    def unskewed(self) -> Relation:
        mat = copy.deepcopy(self.relation.mat)
        for r in mat:
            if r[self.arraySpec.spaceDims] != 0:
                for i in range(self.arraySpec.spaceDims):
                    r[i] = 0
        return Relation(self.relation.inDom, self.relation.inIndices, self.relation.outDom, mat)


class DataflowError(RuntimeError):
    def __init__(self, *args, **kwargs):
        super(DataflowError, self).__init__(*args, **kwargs)


class Dataflow:
    def __init__(
            self,
            arraySpec: ArraySpec,
            opSpec: OpSpec,
            accEntries: List[AccessEntry],
            invDataflow: np.ndarray,
            varSet: VarSet,
            spaceSel: List[Tuple[str, int]],
            timeSel: List[Tuple[str, int]],
            skewCoefs: List[int],
            spaceRange: List[int],
            dataLayouts: List[DataLayout] = []
    ):
        self.arraySpec = arraySpec
        self.opSpec = opSpec

        self.accEntries = accEntries
        self.invDataflow = invDataflow
        self.varSet = varSet
        self.spaceSel = spaceSel
        self.timeSel = timeSel

        self.spaceDims = arraySpec.spaceDims
        self.timeDims = len(timeSel)
        self.skewCoefs = skewCoefs
        self.spaceRange = spaceRange

        if len(dataLayouts) == 0:
            indices = self.arraySpec.spaceNames + \
                [f"t{i+1}" for i in range(self.timeDims)]
            self.dataLayouts = [
                extractLayout(arraySpec, "E", indices,
                              self.timeDims, invDataflow, entry, tensor.accFunc)
                for (tensor, entry) in zip(opSpec.tensors, accEntries)]
        else:
            self.dataLayouts = dataLayouts

        self.timeRange = [
            min(self.varSet.varCoefs[var][i + 1], self.opSpec.getIterator(var).getSize()) if i + 1 < len(self.varSet.varCoefs[var]) else upDiv(
                self.opSpec.getIterator(var).getSize(), self.varSet.varCoefs[var][-1])
            for (var, i) in self.timeSel]

        self.skewedT1 = self.timeRange[0] - \
            int(np.inner(self.skewCoefs,np.add(self.spaceRange,-1)))

    def permuteOuter(self) -> Iterator['Dataflow']:
        # TODO: implement this
        pass

    def exactAccessEntry(self):
        result = []
        for tensor, entry in zip(self.opSpec.tensors, self.accEntries):
            m = tensor.accFunc.mat @ self.invDataflow[:,:self.arraySpec.spaceDims+1]
            print(m)
            try:
                e = self.arraySpec.lookUpEntry(m)
                result.append(e.name)
            except DecomposeError:
                result.append("Unknown")
        return result

    @staticmethod
    def makeDataflow(arraySpec: ArraySpec, opSpec: OpSpec, accEntries: List[AccessEntry], dataLayouts: List[DataLayout]):
        timeDims = max([l.inDims["time"] for l in dataLayouts])

        r = reduce(lambda x, y: x.stack(y, "T"),
                   [e.extendRelation(timeDims) @ l.relation
                    for e, l in zip(accEntries, dataLayouts)])

        a = reduce(lambda x, y: x.stack(y, "T"),
                   [t.accFunc for t in opSpec.tensors])

        mat = np.linalg.pinv(a.mat) @ r.mat
        # print(mat)
        if not np.allclose(a.mat @ mat, r.mat) or np.linalg.matrix_rank(mat) < mat.shape[0]:
            raise DataflowError("Invalid Dataflow")

        intMat = np.round(mat)
        if np.linalg.norm(mat - intMat) > 1e-5:
            raise DataflowError("Invalid Dataflow")
        mat = np.array(intMat, int)

        varSet = VarSet({i.name: [] for i in opSpec.iterators})
        spaceSel = [None for i in range(arraySpec.spaceDims)]
        timeSel = [None for i in range(timeDims)]

        for i, row in zip(opSpec.iterators, mat):
            coefs = []
            for j, v in enumerate(row):
                if v > 0:
                    coefs.append(v)
                    if j < arraySpec.spaceDims:
                        spaceSel[j] = (i.name, v)
                    else:
                        timeSel[j - arraySpec.spaceDims] = (i.name, v)
            varSet.varCoefs[i.name] = sorted(coefs)
        return Dataflow(
            arraySpec=arraySpec,
            opSpec=opSpec,
            accEntries=accEntries,
            invDataflow=mat,
            varSet=varSet,
            spaceSel=spaceSel,
            timeSel=timeSel,
            dataLayouts=dataLayouts
        )

    def bufferSize(self, bufDim=1):
        with open("memEst.in", "w") as fout:
            indices = self.arraySpec.spaceNames + \
                [f"t{i+1}" for i in range(self.timeDims)]
            ranges = list(self.spaceRange) + self.timeRange
            for i in range(self.arraySpec.spaceDims + bufDim, len(ranges)):
                ranges[i] = 1
            fout.write(
                "{" + f"E[{','.join(indices)}]:{' and '.join(map(lambda x: f'0<={x[0]}<{x[1]}' ,zip(indices,ranges)))}" + "}\n")
            for (t, l) in zip(self.opSpec.tensors, self.dataLayouts):
                fout.write(t.domainStr() + "\n" + str(l.unskewed()) + "\n")
        try:
            subprocess.run(["./bin/memEst",
                            "memEst.in",
                            "memEst.out",
                            str(len(self.dataLayouts))], timeout=5)
        except subprocess.TimeoutExpired:
            return MAXBUFFERSIZE

        with open("memEst.out", "r") as fin:
            lines = fin.readlines()

        os.remove("memEst.in")
        os.remove("memEst.out")

        return sum([int(l.strip()) for l in lines])

    def tileTime(self, bufDim=1):
        return int(np.product(self.timeRange[:bufDim]))

    def peakLatency(self):
        return int(np.product(self.timeRange))


def extractLayout(
        arraySpec: ArraySpec,
        inDom: str,
        inIndices: List[str],
        timeDims: int,
        invDataflow: np.ndarray,
        entry: AccessEntry,
        accFunc: Relation):
    m = accFunc.mat @ invDataflow @ np.linalg.pinv(
        entry.extendRelation(timeDims).mat)
    m = np.array(np.round(m), int)
    return DataLayout(
        arraySpec=arraySpec,
        relation=Relation(
            inDom=inDom,
            inIndices=inIndices,
            outDom=accFunc.outDom,
            mat=m
        )
    )
