import numpy as np


class DecomposeError(RuntimeError):
    def __init__(self, what):
        super(DecomposeError, self).__init__(what)


class EntryExtracter:
    def __init__(self, entryConfig):
        self.entryConfig = entryConfig

        self.dirVecs = entryConfig["dirVecs"]
        for i, v in enumerate(self.dirVecs):
            self.dirVecs[i] = np.array(v)

        self.entries = entryConfig["entries"]
        self.vecSets = {}
        for e in self.entries:
            self.vecSets[e["name"]] = set()
            for row in e["vecs"]:
                self.vecSets[e["name"]].add(self.findVec(row))

        self.spaceDims = entryConfig["spaceDims"]

    def findVec(self, vec):
        for i, v in enumerate(self.dirVecs):
            if np.all(vec == v):
                return i
        raise DecomposeError("Undefined dir-vec " + str(vec) +
                             " in entry " + self.entryConfig["name"])

    def __call__(self, movement):
        mat = np.array(movement)
        r = self.spaceDims + 1 - np.linalg.matrix_rank(mat)
        found = set()
        for i, v in enumerate(self.dirVecs):
            if np.allclose(mat @ v, np.zeros(mat.shape[0])):
                found.add(i)

        notEnough = False
        for e in self.entries:
            if self.vecSets[e["name"]] >= found:
                if len(self.vecSets[e["name"]]) == r:
                    return e
                else:
                    notEnough = True
        if notEnough:
            raise DecomposeError(
                "Potential undefined dir-vec in movement " + str(mat))
        else:
            raise DecomposeError(
                "No matching data entry for movemnet " + str(mat))


def extractLayout(dataflow, entry, accFunc):
    m = accFunc @ np.linalg.inv(dataflow)
    if not isinstance(entry, np.ndarray):
        entry = np.array(entry)
    n = m.shape[1] - entry.shape[1]
    entry = np.vstack([
        np.hstack([entry, np.zeros((entry.shape[0], n))]),
        np.hstack([np.zeros((n, entry.shape[1])), np.eye(n)])
    ])
    return m @ np.linalg.pinv(entry)
