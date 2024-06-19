import functools
import itertools
from typing import List, Union, Optional, Iterator, Set
from collections import deque
import time

import numpy as np
import diophantine

from rubick.relation import *
from rubick.ir import *
from rubick.util import upDiv


class DataflowDSE:
    def __init__(self, arraySpec: ArraySpec):
        self.arraySpec = arraySpec

        # self.spaceRangePower = []
        # r = 1
        # for i in range(self.arraySpec.spaceDims + 2):
        #     self.spaceRangePower.append(r)
        #     r *= self.arraySpec.spaceRange

    def __call__(self, opSpecs: List[OpSpec], permuteOuter: Optional[int] = None, exactReuse: bool = False) -> Iterator[Tuple[List[AccessEntry], List[Iterator[Dataflow]]]]:
        numTensors = len(opSpecs[0].tensors)
        for opSpec in opSpecs:
            if len(opSpec.tensors) != numTensors:
                raise RuntimeError(
                    "DataflowDSE: Operatos have different number of tensors!")
        for accEntries in self._iterAccEntries(numTensors):
            yield (accEntries, [self._iterDataflow(opSpec, permuteOuter, exactReuse, accEntries) for opSpec in opSpecs])

    def _iterDataflow(self, opSpec: OpSpec, permuteOuter: Optional[int], exactReuse: bool, entries: List[AccessEntry]):
        for invDataflow, spaceSel, timeSel, varSet, spaceRange in self._iterInvDataflow(opSpec, entries):
            if exactReuse and not self._checkExactReuse(opSpec, invDataflow, entries):
                continue
            # print(spaceSel)
            # print([opSpec.getIterator(i).getSize() >= self.arraySpec.spaceRange
            #        for i in spaceSel]
            #       )
            skewCoefs = invDataflow[timeSel][0:self.arraySpec.spaceDims]
            spaceSel = [(opSpec.iterators[s].name, varSet.findID(opSpec.iterators[s].name, invDataflow[s][i]))
                        for i, s in enumerate(spaceSel)]
            timeDims = 1
            timeSel = [(opSpec.iterators[timeSel].name,
                        varSet.findID(
                            opSpec.iterators[timeSel].name, invDataflow[timeSel][self.arraySpec.spaceDims])
                        )]

            for (u, v) in varSet.varCoefs.items():
                if len(v) > 1 and v[-1] == v[-2]:
                    del v[-1]
                # while len(v) > 1 and v[-1] >= opSpec.getIterator(u).getSize():
                    # del v[-1]
            
            
            name2row = {}
            temp = []
            for i, (u, v) in enumerate(varSet.varCoefs.items()):
                if (u, len(v)-1) not in timeSel:
                    temp.append((u, len(v) - 1))
                    timeDims += 1
                    name2row[u] = i

            temp.sort(key=lambda x: varSet.varCoefs[x[0]][x[1]])
            timeSel += temp
            # print([(u, varSet.varCoefs[u][v])for (u,v) in timeSel])

            def makeDataflow(timeSel):
                m = np.zeros((invDataflow.shape[0], timeDims - 1))
                for i, (u, v) in enumerate(timeSel[1:]):
                    m[name2row[u]][i] = varSet.varCoefs[u][v]
                newInvDataflow = np.hstack((invDataflow, m))
                dataflow = Dataflow(self.arraySpec, opSpec, entries,
                                    newInvDataflow, varSet, spaceSel, timeSel, skewCoefs, spaceRange)
                # print(dataflow)
                return dataflow

            if permuteOuter is not None and permuteOuter > 1:
                for head in itertools.permutations(timeSel[1:], min(permuteOuter - 1, len(timeSel) - 1)):
                    tail = [i for i in timeSel[1:] if i not in head]
                    newTimeSel = [timeSel[0]] + list(head) + tail
                    yield makeDataflow(newTimeSel)
            else:
                yield makeDataflow(timeSel)

    def _iterAccEntries(self, n):
        def rankCheck(entries: List[AccessEntry]):
            a = functools.reduce(lambda x, y: x.stack(
                y, "T"), map(lambda x: x.relation, entries))
            return np.linalg.matrix_rank(a.mat) == self.arraySpec.spaceDims + 1

        def dirCheck(entries: List[AccessEntry]):
            sysVecs = set()
            mulVecs = set()
            for e in entries:
                if not e.stationary:
                    sysVecs.update(e.systolic)
                    mulVecs.update(e.multicast)
            return len(sysVecs & mulVecs) == 0

        def ioCheck(entries: List[AccessEntry]):
            return reduce(lambda x, y: x and y, [e.canInput for e in entries[:-1]] + [entries[-1].canOutput])

        def validCheck(entries: List[AccessEntry]):
            return rankCheck(entries) and dirCheck(entries) and ioCheck(entries)
        return filter(
            validCheck,
            itertools.product(*itertools.repeat(self.arraySpec.entries, n))
        )

    def _iterInvDataflow(self, opSpec: OpSpec, entries: List[AccessEntry]) -> Iterator[Tuple[np.ndarray, List[int], int, VarSet, List[int]]]:
        constraints = self._collectConstraints(opSpec, entries)

        n = opSpec.numIter
        m = self.arraySpec.spaceDims + 1

        constraints, spaceChoices, timeChoices = self._nonZeroPropagate(
            n, constraints)

        # print(spaceChoices)
        # spaceChoices = [
        #     [j for j in i if opSpec.getIterator(
        #         j).getSize() >= self.arraySpec.spaceRange]
        #     for i in spaceChoices]
        # print(spaceChoices)
        # print([[j for j in i if opSpec.getIterator(j).getSize() >= self.arraySpec.spaceRange]
        #    for i in spaceChoices])

        for spaceSel in itertools.product(*spaceChoices):
            for timeSel in timeChoices:
                if self._checkImpossible(constraints, spaceSel, timeSel):
                    continue
                for spaceRange in itertools.product(self.arraySpec.spaceRange, repeat=self.arraySpec.spaceDims):
                    coefs = self._collectCoefs(opSpec, entries, spaceRange)
                    for rows in itertools.product(*[self._iterCoefs(i, spaceSel, timeSel, coefs, opSpec.getIterator(i).getSize(), spaceRange) for i in range(n)]):
                        mat = np.zeros((n, m))
                        for i, (r, c) in enumerate(rows):
                            for u, v in r.items():
                                mat[i][u] = v

                        if findTrue(constraints, lambda c: not c.check(mat)):
                            mat = self._solve(mat, spaceChoices,
                                            timeSel, constraints)
                            if mat is None:
                                continue

                        varSet = VarSet(
                            {i.name: c for i, (r, c) in zip(opSpec.iterators, rows)})

                        yield mat, spaceSel, timeSel, varSet, spaceRange

    def _iterCoefs(self, rowId:int, spaceSel:List[int], timeSel:int, coefs: List[int], limit:int, spaceRange:list[int]) -> Iterator[Tuple[Dict[int, int], List[int]]]:
        dims = [i for i, s in enumerate(spaceSel) if s == rowId]
        timeDim = self.arraySpec.spaceDims
        if rowId == timeSel:
            dims += [timeDim]
        for order in itertools.permutations(dims):
            for c in itertools.product(*[
                [spaceRange[i]] if i < timeDim
                else ([1] if order[-1] == timeDim else coefs)
                for i in order
            ]):
                m = {}
                varCoefs = []
                r = 1
                for i, x in zip(order, c):
                    if r>limit:
                        r = limit+1
                        break
                    varCoefs.append(r)
                    m[i] = r
                    r *= x
                    if i == timeDim and order[-1] == timeDim:
                        if limit <= r:
                            r = limit+1
                        else:
                            r = limit
                if r>limit:
                    continue
                if r!=limit:
                    varCoefs.append(r)
                yield m, varCoefs

        # spaceDims = [i for i, s in enumerate(spaceSel) if s == rowId]
        # if rowId == timeSel:
        #     return self._iterCoefsWithTime(spaceDims, coefs, limit)
        # else:
        #     return self._iterCoefsWithoutTime(spaceDims, limit)
        

    def _iterCoefsWithTime(self, spaceDims, coefs, limit) -> Iterator[Tuple[Dict[int, int], List[int]]]:
        timeDim = self.arraySpec.spaceDims
        for order in itertools.permutations(spaceDims + [timeDim]):
            for timeCoef in ([1] if order[-1] == spaceDims else coefs):
                m = {}
                varCoefs = []
                r = 1
                for i in order:
                    if r>limit:
                        r = -1
                        break
                    varCoefs.append(r)
                    m[i] = r
                    if i == timeDim:
                        r *= timeCoef
                    else:
                        r *= self.arraySpec.spaceRange
                if r == -1:
                    continue
                if r<limit:
                    varCoefs.append(r)
                yield m, varCoefs

    def _iterCoefsWithoutTime(self, spaceDims, limit) -> Iterator[Tuple[Dict[int, int], List[int]]]:
        if len(spaceDims) == 0:
            yield dict(), [1]
        else:
            for order in itertools.permutations(spaceDims):
                if self.spaceRangePower[len(spaceDims)] <= limit:
                    yield {i: r for i, r in zip(order, self.spaceRangePower)}, self.spaceRangePower[:len(spaceDims) + 1]

    def _collectCoefs(self, opSpec: OpSpec, entries: List[AccessEntry], spaceRange:List[int]) -> List[int]:
        '''Collec the possible coefficients for t1 in the inverse dataflow matrix'''
        systolic = functools.reduce(
            lambda x, y: x or y, map(lambda e: len(e.systolic) > 0, entries))
        coefs = set(spaceRange)
        for t in opSpec.tensors:
            for row in t.accFunc.mat:
                for i in row:
                    if i == 0:
                        continue
                    coefs.add(i)
                    if (systolic):
                        for x in spaceRange:
                            coefs.add(x * i)
                            if x % i == 0:
                                coefs.add(x / i)
        if 1 in coefs:
            coefs.remove(1)
        return coefs

    def _collectConstraints(self, opSpec: OpSpec, entries: List[AccessEntry]) -> List['InvDataflowConstraint']:
        result = []
        for tensor, entry in zip(opSpec.tensors, entries):
            for v in entry.vecs:
                for row in tensor.accFunc.mat:
                    result.append(InvDataflowConstraint(np.outer(row, v)))
        return result

    def _solve(self, mat: np.ndarray, spaceChoices: List[List[int]], timeSel: int, constraints: List['InvDataflowConstraint']) -> np.ndarray:
        '''Solve the skew coefficients for the systolic dataflows'''
        b = np.array([-np.sum(mat * c.mat) for c in constraints])
        rows = []
        ids = []
        for i in range(self.arraySpec.spaceDims):
            if mat[timeSel][i] == 0 and timeSel in spaceChoices[i]:
                ids.append(i)
                rows.append([c.mat[timeSel][i] for c in constraints])

        if len(rows) == 0:
            return None

        a = np.array(rows).T
        try:
            x = diophantine.solve(a, b)
            x = np.array(x).flatten()
            if len(x) == 0:
                return None
        except NotImplementedError:
            x = np.linalg.pinv(a) @ b
        except diophantine.NoSolutionException:
            return None

        for i, v in zip(ids, x):
            if abs(v - round(v)) > 1e-5:
                return None
            mat[timeSel][i] = int(round(v))
        return mat

    def _nonZeroPropagate(self, n: int, constraints: List['InvDataflowConstraint']) -> Tuple[List['InvDataflowConstraint'], List[List[int]], List[int]]:
        '''
        Find all constraint matrices that has only one non-zero.
        If a constraint matrix has only one non-zero, that position of the inverse dataflow matrix should be zero.
        Other constraint matrices that has a non-zero at that position should remove the non-zero.
        '''
        nonZeros = []
        nonZerosCnt = []
        removed = [False] * len(constraints)
        pos2id = {}
        zeros = set()
        for i, c in enumerate(constraints):
            nonZeros.append([])
            nonZerosCnt.append(len(c.nonZeros))
            for (x, y, _) in c.nonZeros:
                nonZeros[i].append((x, y))
                if (x, y) in pos2id:
                    pos2id[(x, y)].append(i)
                else:
                    pos2id[(x, y)] = [i]

        que = deque()
        for i, v in enumerate(nonZerosCnt):
            if v == 1:
                que.append(i)
                removed[i] = True
            elif v == 0:
                removed[i] = True

        while (len(que) > 0):
            i = que.popleft()
            if nonZerosCnt[i] == 0:
                continue
            x, y = None, None
            for x, y in nonZeros[i]:
                if (x, y) not in zeros:
                    break
            zeros.add((x, y))

            for j in pos2id[(x, y)]:
                nonZerosCnt[j] -= 1
                if nonZerosCnt[j] <= 1 and not removed[j]:
                    que.append(j)
                    removed[j] = True

        newConstraints = []
        for i, c in enumerate(constraints):
            if not removed[i] and not findTrue(newConstraints, lambda c1: np.all(c1.mat == c.mat)):
                newConstraints.append(c)
        spaceChoices = [
            list(filter(lambda x: (x, i) not in zeros, range(n)))
            for i in range(self.arraySpec.spaceDims)
        ]
        timeChoices = list(
            filter(lambda x: (x, self.arraySpec.spaceDims) not in zeros, range(n)))

        return newConstraints, spaceChoices, timeChoices

    def _checkImpossible(self, constraints: List['InvDataflowConstraint'], spaceSel: List[int], timeSel: int) -> bool:
        '''
        If all non-zeros in the constraint matrix are the same or all possive,
        then the non-zeros in the inverse matrix should be checked so that it's possible to satisfy the constraints.
        '''
        positive = {(j, i) for i, j in enumerate(spaceSel)}
        positive.add((timeSel, self.arraySpec.spaceDims))

        free = {(timeSel, i) for i in range(self.arraySpec.spaceDims)
                if (timeSel, i) not in positive}

        for c in constraints:
            if (c.same or c.allPositive) and len(positive & c.nonZeroPos) and len(free & c.nonZeroPos) == 0:
                return True
        return False

    def _checkExactReuse(self, opSpec: OpSpec, invDataflow: np.ndarray, entries: List[AccessEntry]):
        for tensor, entry in zip(opSpec.tensors, entries):
            m = tensor.accFunc.mat @ invDataflow
            try:
                e = self.arraySpec.lookUpEntry(m)
            except DecomposeError:
                return False
            if e.name != entry.name:
                return False
        return True


class InvDataflowConstraint:
    def __init__(self, mat: np.ndarray):
        '''The point-wise product of the inverse dataflow matrix and the constraint matrix should be a zero matrix'''
        self.mat = mat
        self.nonZeros = []
        self.nonZeroPos = set()
        self.allPositive = True
        self.same = True
        last = None
        for i in range(self.mat.shape[0]):
            for j in range(self.mat.shape[1]):
                if self.mat[i][j] != 0:
                    self.nonZeros.append([i, j, self.mat[i][j]])
                    self.nonZeroPos.add((i, j))
                    if self.mat[i][j] < 0:
                        self.allPositive = False
                    if last is not None and self.mat[i][j] != last:
                        self.same = False
                    last = self.mat[i][j]

    def check(self, invDataflow):
        return abs(np.sum(self.mat * invDataflow)) < 1e-5


def findTrue(x: List, pred) -> bool:
    '''Check if any pred(x[i]) is True'''
    for i in x:
        if pred(i):
            return True
    return False
