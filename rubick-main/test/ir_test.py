import numpy as np
from rubick.ir import *
from rubick.relation import *

if __name__ == "__main__":
    arraySpec = ArraySpec("data/2D_entry.json")

    opSpec = OpSpec()

    i, j, k = opSpec.genIterators(3)
    A, B, C = opSpec.genTensors(3)

    i.setRange(0, 512)
    j.setRange(0, 512)
    k.setRange(0, 512)

    A.setRange(512, 512)
    B.setRange(512, 512)
    C.setRange(512, 512)

    opSpec.setExpr(C[i][j], A[i][k] * B[k][j])

    layoutA = DataLayout(
        arraySpec,
        "E[x,y,t1,t2,t3]->A[y+8*t2, -y+t1]"
    )
    layoutB = DataLayout(
        arraySpec,
        "E[x,y,t1,t2,t3]->B[-x+t1, x+8*t3]"
    )
    layoutC = DataLayout(
        arraySpec,
        "E[x,y,t1,t2,t3]->B[y+8*t2, x+8*t3]"
    )
    print(opSpec.getTensor('A').accFunc)
    print(opSpec.getTensor('B').accFunc)
    print(opSpec.getTensor('C').accFunc)
    dataflow = Dataflow.makeDataflow(
        arraySpec,
        opSpec,
        [arraySpec.getEntry("X-systolic"),
         arraySpec.getEntry("Y-systolic"),
         arraySpec.getEntry("Stationary")],
        [layoutA, layoutB, layoutC]
    )

    print(dataflow.varSet)
    print(dataflow.spaceSel)
    print(dataflow.timeSel)
