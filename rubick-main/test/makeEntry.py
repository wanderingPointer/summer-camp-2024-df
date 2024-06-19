import json

def makeEntry(name, relation, vecs):
    return {
        "name":name,
        "relation":relation,
        "vecs":vecs,
        "hasDiag": False,
        "input": True,
        "output": True
    }

if __name__ == "__main__":
    result = []

    sysVec = {
        "x": [1,0,0,1],
        "y": [0,1,0,1],
        "z": [0,0,1,1]
    }
    mulVec = {
        "x": [1,0,0,0],
        "y": [0,1,0,0],
        "z": [0,0,1,0]
    }
    staVec = [0,0,0,1]

    for a in ['x','y','z']:
        for b in ['x','y','z']:
            if a != b:
                name = a.upper()+"-systotic-"+b.upper()+"-multicast"
                relation = "D[x,y,z,t1]->E["+",".join(['0' if c==a or c==b else c for c in ['x','y','z']])+",t1-"+a+"]"
                vecs = [
                    sysVec[a],
                    mulVec[b]
                ]
                result.append(makeEntry(name, relation, vecs))
    
    for a in ['x','y','z']:
        name = a.upper()+"-multicast-stationary"
        relation = "D[x,y,z,t1]->E["+",".join(['0' if c==a else c for c in ['x','y','z']])+",0]"
        vecs = [
            mulVec[a],
            staVec
        ]
        result.append(makeEntry(name, relation, vecs))

    for a in ['x','y','z']:
        for b in ['x','y','z']:
            if a < b:
                name = a.upper()+b.upper()+"-multicast-stationary"
                relation = "D[x,y,z,t1]->E["+",".join(['0' if c==a or c==b else c for c in ['x','y','z']])+",0]"
                vecs = [
                    mulVec[a],
                    mulVec[b],
                    staVec
                ]
                result.append(makeEntry(name, relation, vecs))
    
    with open("makeEntry.json","w") as fout:
        json.dump([result], fout, indent=2)