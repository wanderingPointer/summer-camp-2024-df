import numpy as np
from rubick.relation import Index, Relation, InvTilingRelation

if __name__ == "__main__":
    a = Relation.read(
        "S[i,j, k]->D[j * 5,k%8,2*(i +j%8) *2 +3 *k%8, j/8, k/8]")
    print(a)
    print(a.toStr(" ", False, True))
    print(a.mat)

    a = Relation.read("D[x,y,t1]->E[0,y,t1-(x+y)]")
    print(a)
    print(a.mat)

    a = Relation.read("{E[x y t1]->A[y x]}", sep=" ")
    print(a)

    a = InvTilingRelation("S", [("i", 8), ("j", 8), ("k", 1)])
    print(a)
    b = Relation.read("S[i,j,k]->A[i+k,j]")
    print(a.mat)
    print(b.mat)
    c = a @ b
    print(c)
