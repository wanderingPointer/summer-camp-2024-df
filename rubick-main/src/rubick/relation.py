from collections import deque
import copy
from typing import List, Dict, Mapping, Optional, Iterator, Union, Tuple

import numpy as np


class VarSet:
    def __init__(self, varCoefs: Dict[str, List[int]]):
        # var = var0*coef0 + var1*coef1 + var2*coef2 + ...
        self.varCoefs = varCoefs
        self.pos = {i: []for i in self.varCoefs.keys()}

        cnt = 0
        for u, v in varCoefs.items():
            for i in range(1, len(v)):
                if v[i] <= v[i - 1]:
                    raise RuntimeError("Illegal coefs")
            if len(v) == 1 and v[0] != 1:
                raise RuntimeError("Illegal coefs")
            for i in range(len(v)):
                self.pos[u].append(cnt)
                cnt += 1

    def findID(self, name: str, coef: int):
        return self.varCoefs[name].index(coef)

    def __str__(self) -> str:
        return str(self.varCoefs)


class Relation:
    def __init__(self, inDom: str, inIndices: List[str], outDom: str, mat: np.ndarray):
        self.inDom = inDom
        self.inIndices = inIndices
        self.outDom = outDom
        self.mat = mat

        if self.mat.shape[1] != len(self.inIndices):
            raise RuntimeError(
                "Relation: incompatible matrix and input indices")

    def __matmul__(self, other):
        return self.chain(other)

    def chain(self, other):
        if not isinstance(other, Relation):
            raise TypeError(
                "Relation: the operands of chain should be another relation")
        if self.outDom != other.inDom:
            raise RuntimeError(
                "Relation: chaining two relations of different domains")

        return Relation(
            inDom=self.inDom,
            inIndices=self.inIndices,
            outDom=other.outDom,
            mat=other.mat @ self.mat
        )

    def stack(self, other, outDom):
        if not isinstance(other, Relation):
            raise TypeError(
                "Relation: the operands of chain should be another relation")
        if self.inDom != other.inDom:
            raise RuntimeError(
                "Relation: concatenating two relations of different domains")
        if self.inIndices != other.inIndices:
            raise NotImplemented

        return Relation(
            inDom=self.inDom,
            inIndices=self.inIndices,
            outDom=outDom,
            mat=np.vstack((self.mat, other.mat))
        )

    def toStr(self, sep=", ", haveBracket=True, toLatex=False):
        left = self.inDom + \
            "[" + sep.join(self.inIndices) + "]"

        mulOp = r"$\cdot$" if toLatex else "*"

        def makeItem(k, i):
            if k == 1:
                return f"+{i}"
            elif k == -1:
                return f"-{i}"
            elif k > 0:
                return f"+{k}{mulOp}{i}"
            elif k < 0:
                return f"{k}{mulOp}{i}"
            else:
                return ""

        def trim(s):
            if s == "":
                return "0"
            elif s[0] == '+':
                return s[1:]
            else:
                return s
        right = self.outDom + \
            "[" + sep.join([
                trim("".join([makeItem(k, i)
                     for k, i in zip(r, self.inIndices)]))
                for r in self.mat
            ]) + "]"

        s = left + (r"$\to$" if toLatex else "->") + right
        if haveBracket:
            if toLatex:
                s = r"\{" + s + r"\}"
            else:
                s = "{" + s + "}"
        return s

    def __str__(self) -> str:
        return self.toStr()

    def changeDomain(self, inDom: str, outDom: str) -> 'Relation':
        return Relation(inDom=inDom, inIndices=self.inIndices, outDom=outDom, mat=self.mat)

    @staticmethod
    def read(expr: str, sep: str = ','):
        x = expr.strip()
        if x[0] == '{' and x[-1] == '}':
            x = x[1:-1]

        left, right = Relation._trySplit2(x, "->", expr, "relation")
        inDom, inIndices = Relation.readLeft(left, sep)
        outDom, mat = Relation.readRight(inIndices, right, sep)
        return Relation(inDom=inDom, inIndices=inIndices, outDom=outDom, mat=mat)

    @staticmethod
    def readLeft(expr: str, sep: str = ','):
        inDom, rest = Relation._trySplit2(expr, '[', expr, "left expression")
        x = Relation._trySplit2(rest, ']', expr, "left expression")
        inIndices = [i.strip() for i in x[0].strip().split(sep)]
        return inDom, inIndices

    @staticmethod
    def readRight(inIndices: List[str], expr: str, sep: str = ','):
        outDom, rest = Relation._trySplit2(expr, '[', expr, "right expression")
        x = Relation._trySplit2(rest, ']', expr, "right expression")
        items = x[0].strip().split(sep)
        items = [Relation._parseExpr(i.strip()) for i in items]

        indiceIDs = {x: i for i, x in enumerate(inIndices)}

        # transform
        rows = []
        for i in items:
            row = [0] * len(inIndices)
            for u, v in i.items():
                row[indiceIDs[u]] += v
            rows.append(row)
        mat = np.array(rows)
        return outDom, mat

    @staticmethod
    def _trySplit2(expr: str, sep: str, toShow: str, errType: str):
        items = expr.strip().split(sep)
        if len(items) != 2:
            raise RuntimeError(
                f"Relation: {toShow} is not a valid {errType}"
            )
        return items

    @staticmethod
    def _parseExpr(expr: str) -> Dict[str, int]:
        if expr == "0":
            return dict()
        stack = deque(['('])

        def throw():
            raise RuntimeError(f"Relation: {expr} is not a valid expression")

        def mergeAdd(a, b):
            c = copy.copy(a)
            for u, v in b.items():
                if u in c:
                    c[u] += v
                else:
                    c[u] = v
            return c

        def mergeSub(a, b):
            c = copy.copy(a)
            for u, v in b.items():
                if u in c:
                    c[u] -= v
                else:
                    c[u] = -v
            return c

        def mergeMul(m, k):
            return {
                u: v * k for u, v in m.items()
            }

        def popStack():
            a = stack.pop()
            if not isinstance(a, (dict, int)):
                throw()
            op = stack.pop()
            if not isinstance(op, str) and op not in ["*", "+", "-"]:
                throw()
            if not isinstance(stack[-1], (dict, int)):
                if op != '-':
                    throw()
                if isinstance(a, int):
                    stack.append(-a)
                else:
                    stack.append({u: -v for u, v in a.items()})
                return

            b = stack.pop()
            if op == '+':
                if isinstance(a, int):
                    if isinstance(b, int):
                        c = a + b
                    else:
                        raise RuntimeError(
                            f"Relation: {expr} has a constant which is not supported yet")
                else:
                    if isinstance(b, int):
                        raise RuntimeError(
                            f"Relation: {expr} has a constant which is not supported yet")
                    else:
                        c = mergeAdd(a, b)
            elif op == '-':
                if isinstance(a, int):
                    if isinstance(b, int):
                        c = b - a
                    else:
                        raise RuntimeError(
                            f"Relation: {expr} has a constant which is not supported yet")
                else:
                    if isinstance(b, int):
                        raise RuntimeError(
                            f"Relation: {expr} has a constant which is not supported yet")
                    else:
                        c = mergeSub(b, a)
            else:
                if isinstance(a, int):
                    if isinstance(b, int):
                        c = a * b
                    else:
                        c = mergeMul(b, a)
                else:
                    if isinstance(b, int):
                        c = mergeMul(a, b)
                    else:
                        throw()
            stack.append(c)

        x = expr + ')'
        for c in Relation._lex(x):
            if isinstance(c, int):
                if isinstance(stack[-1], (dict, int)):
                    throw()
                stack.append(c)
                if len(stack) > 1 and stack[-1] == '-' and not isinstance(stack[-2], (dict, int)):
                    popStack()
            elif c == '(':
                stack.append(c)
            elif c == ')':
                while (len(stack) > 1 and stack[-2] != '('):
                    popStack()
                v = stack.pop()
                stack.pop()
                stack.append(v)
            elif c == '+':
                while (len(stack) > 1 and stack[-2] != '('):
                    popStack()
                stack.append(c)
            elif c == '-':
                while (len(stack) > 1 and stack[-2] != '('):
                    popStack()
                stack.append(c)
            elif c == '*':
                while (len(stack) > 1 and stack[-2] == '*'):
                    popStack()
                stack.append(c)
            else:
                if isinstance(stack[-1], (dict, int)):
                    throw()
                stack.append({c: 1})
                if len(stack) > 1 and stack[-1] == '-' and not isinstance(stack[-2], (dict, int)):
                    popStack()
        v = stack.pop()
        if len(stack) != 0 or not isinstance(v, dict):
            throw()
        return v

    @staticmethod
    def _lex(expr: str) -> Iterator[Union[str, int]]:
        token = ""
        state = "empty"
        spaced = False

        def throw():
            raise RuntimeError(f"Relation: {expr} is not a valid item")

        def yieldToken():
            nonlocal state
            nonlocal token
            if state == 'name':
                rst = token
            elif state == 'num':
                rst = int(token)
            else:
                raise RuntimeError(f"Relation: internal error on {expr}")

            state = 'empty'
            token = ''
            return rst

        for c in expr:
            if c == ' ':
                spaced = True
            else:
                # Automata
                if c == '*' or c == '+' or c == '-' or c == '(' or c == ')':
                    if state != 'empty':
                        yield yieldToken()
                    yield c
                elif c.isalpha() or c == "'":
                    if state == 'empty':
                        state = 'name'
                    elif state != 'name' or spaced:
                        throw()
                    token += c
                elif c.isdigit():
                    if state == 'empty':
                        state = 'num'
                    elif state == 'name' or state == 'num':
                        if spaced:
                            throw()
                    token += c

                spaced = False

        if state != 'empty':
            yield yieldToken()
