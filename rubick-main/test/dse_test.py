import time
from rubick.dataflowDSE import DataflowDSE
from rubick.relation import *
from rubick.ir import *
from rubick.perfModel import PerfModel


def makeCONV1D(s: int):
    opSpec = OpSpec(f"CONV1D-{s}")

    i, j = opSpec.genIterators(2)
    A, B, C = opSpec.genTensors(3)

    i.setRange(0, 512)
    j.setRange(0, 64)

    A.setRange(512)
    B.setRange(64)
    C.setRange(512)

    opSpec.setExpr(C[i], A[i + s * j] * B[j])
    return opSpec


def makeGEMM():
    opSpec = OpSpec("GEMM")

    i, j, k = opSpec.genIterators(3)
    A, B, C = opSpec.genTensors(3)

    i.setRange(0, 512)
    j.setRange(0, 512)
    k.setRange(0, 512)

    A.setRange(512, 512)
    B.setRange(512, 512)
    C.setRange(512, 512)

    opSpec.setExpr(C[i][j], A[i][k] * B[k][j])
    return opSpec


def makeCONV():
    opSpec = OpSpec("CONV")

    k, c, ox, oy, rx, ry = opSpec.genIterators(6)
    W, I, O = opSpec.genTensors(3)

    k.setRange(0, 512)
    c.setRange(0, 512)
    ox.setRange(0, 7)
    oy.setRange(0, 7)
    rx.setRange(0, 3)
    ry.setRange(0, 3)

    W.setRange(512, 512, 3, 3)
    I.setRange(512, 7, 7)
    O.setRange(512, 7, 7)

    opSpec.setExpr(O[k][ox][oy], W[k][c][rx][ry] * I[c][ox + rx][oy + ry])
    return opSpec


def makeCONV_2():
    opSpec = OpSpec("CONV")

    k, c, ox, oy, rx, ry = opSpec.genIterators(6)
    W, I, O = opSpec.genTensors(3)

    k.setRange(0, 512)
    c.setRange(0, 512)
    ox.setRange(0, 28)
    oy.setRange(0, 28)
    rx.setRange(0, 3)
    ry.setRange(0, 3)

    W.setRange(512, 512, 3, 3)
    I.setRange(512, 28, 28)
    O.setRange(512, 28, 28)

    opSpec.setExpr(O[k][ox][oy], W[k][c][rx][ry] * I[c][ox + rx][oy + ry])
    return opSpec


def makePWCONV():
    opSpec = OpSpec("Pointwise")

    k, c, ox, oy = opSpec.genIterators(4)
    W, I, O = opSpec.genTensors(3)

    k.setRange(0, 256)
    c.setRange(0, 256)
    ox.setRange(0, 64)
    oy.setRange(0, 64)

    W.setRange(256, 256)
    I.setRange(256, 64, 64)
    O.setRange(256, 64, 64)

    opSpec.setExpr(O[k][ox][oy], W[k][c] * I[c][ox][oy])
    return opSpec


def makeDWCONV():
    opSpec = OpSpec("Depthwise")

    c, ox, oy, rx, ry = opSpec.genIterators(5)
    W, I, O = opSpec.genTensors(3)

    c.setRange(0, 256)
    ox.setRange(0, 64)
    oy.setRange(0, 64)
    rx.setRange(0, 3)
    ry.setRange(0, 3)

    W.setRange(256, 3, 3)
    I.setRange(256, 66, 66)
    O.setRange(256, 64, 64)

    opSpec.setExpr(O[c][ox][oy], W[c][rx][ry] * I[c][ox + rx][oy + ry])
    return opSpec


if __name__ == "__main__":
    arraySpec = ArraySpec("data/3D_entry.json")

    t0 = time()
    ops = [makeCONV()]
    perfModel = PerfModel(arraySpec)
    perfModel(ops, 7, 65536, 2.56, False, "cube_conv.json")

    print(time() - t0)
