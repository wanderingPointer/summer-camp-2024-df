from multiprocessing.util import MAXFD
from typing import Iterator, List
import json

import numpy as np
from rubick.dataflowDSE import DataflowDSE

from rubick.relation import *
from rubick.ir import *


class PerfModel:
    def __init__(self, arraySpec):
        self.arraySpec = arraySpec

    def __call__(
        self,
        ops: List[OpSpec],
        maxBufDim: int,
        bufSizeLimit: int,
        bwCapcity: int,
        exactReuse: bool,
        outFile: str
    ):
        dse = DataflowDSE(self.arraySpec)
        with open(outFile, "w") as fout:
            fout.write("[")
            first = True
            for accEntries, dataflowGens in dse(ops, None, exactReuse):
                opDataflow = [[d for d in dataflowGen]
                              for dataflowGen in dataflowGens]
                if not reduce(lambda a, b: a and b, map(lambda x: len(x) > 0, opDataflow)):
                    continue
                curAccEntries = {"accEntries": [
                    str(e) for e in accEntries], "op": {}}

                ok = True
                for op, opD in zip(ops, opDataflow):
                    cur = []
                    for d in opD:
                        lef = 1
                        rig = min(maxBufDim, d.timeDims)
                        while lef != rig:
                            mid = (lef + rig + 1) // 2
                            bufSize = d.bufferSize(mid)
                            if bufSize > bufSizeLimit:
                                rig = mid - 1
                            else:
                                lef = mid
                        bufDim = lef
                        # bufDim = 1
                        bufSize = d.bufferSize(bufDim)
                        if bufSize > bufSizeLimit:
                            continue
                        bwReq = bufSize / d.tileTime(bufDim)
                        latency = d.peakLatency() * max(1.0, bwReq / bwCapcity)
                        # latency = d.peakLatency()

                        cur.append({
                            "dataLayouts": [str(l) for l in d.dataLayouts],
                            "bufDim": bufDim,
                            "bufSize": bufSize,
                            "bwRequirement": bwReq,
                            "latency": latency,
                            "spaceRange": d.spaceRange
                        })
                    if len(cur):
                        curAccEntries["op"][op.name] = cur
                    else:
                        ok = False
                        break

                if ok:
                    if not first:
                        fout.write(",\n")
                    else:
                        first = False
                    fout.write(json.dumps(curAccEntries, indent=2))
            fout.write("]\n")
