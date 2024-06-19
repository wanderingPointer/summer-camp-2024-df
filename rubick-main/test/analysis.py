import json
import heapq
from dataclasses import dataclass, field
from typing import Any
from rubick.relation import *
from rubick.ir import *

topK = 10

@dataclass(order=True)
class Choice:
    priority: int
    item: Any = field(compare=False)

if __name__ == "__main__":
    with open("cube_conv.json", "r") as fin:
        data = json.load(fin)
    buf = []
    lat = []

    pareto = []

    pes2lat = {}

    h = []
    tot = 0
    for curEntries in data:
        entries = curEntries["accEntries"]
        for curDF in curEntries["op"]["CONV"]:
            tot += 1
            layouts = curDF["dataLayouts"]
            choice = Choice(-curDF["latency"], {"entries":entries, "layouts":layouts, "data":curDF})
            if len(h) < topK:
                heapq.heappush(h,choice)
            elif choice > h[0]:
                heapq.heapreplace(h,choice)
            buf.append(curDF["bwRequirement"])
            lat.append(curDF["latency"])

            pareto = [x for x in pareto if x["bwRequirement"]<curDF["bwRequirement"] or x["latency"]<curDF["latency"]]
            if len([x for x in pareto if x["bwRequirement"]<curDF["bwRequirement"] and x["latency"]<curDF["latency"]]) == 0:
                pareto.append(curDF)
            
            pes = curDF["spaceRange"][0] * curDF["spaceRange"][1]
            if pes in pes2lat:
                pes2lat[pes] = min(pes2lat[pes], curDF["latency"])
            else:
                pes2lat[pes] = curDF["latency"]
                
    # with open("speedup.csv", "w") as fout:
        # fout.write("PEs, Latency\n"+"\n".join([f"{x}, {y}"for x, y in pes2lat.items()]))
    # for (u, v) in pes2lat.items():
        # print(u,v)
    # pareto.sort(key=lambda x:x["latency"])
    print(tot)
    for i in h:
        print(i.item)
    # with open("out.csv", "w") as fout:
        # fout.write("bwRequirement, Latency\n"+"\n".join([f"{x}, {y}"for x, y in zip(buf, lat)]))