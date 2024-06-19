import os
import re
import subprocess

import enum


class WRProperty(enum.Enum):
    READ = 0
    WRITE = 1
    READWRITE = 2


class MemoryModel:
    def __init__(self, metric):
        self.metric = metric

    def __call__(self, ports, size):
        area = self.metric["memory"]["areaPerBit"] * \
            size * self.metric["bitWidth"]
        power = self.metric["memory"]["powerPerBit"] * \
            size * self.metric["bitWidth"]
        return area, power


class CACTI(MemoryModel):
    def __init__(self, metric):
        super(CACTI, self).__init__(metric)
        self.cacti = self.metric["memory"]["CACTI"]

        with open(self.cacti["cfgTemplate"], "r") as fin:
            self.cfgTemplate = "".join(fin.readlines())

        self.pattern = {
            "area": re.compile("Cache height x width \(mm\): (?P<width>[0-9.]+) x (?P<height>[0-9.]+)"),
            "cycle": re.compile("Cycle time \(ns\):  (?P<cycle>[0-9.]+)"),
            "readE": re.compile("Total dynamic read energy per access \(nJ\): (?P<energy>[0-9.]+)"),
            "writeE": re.compile("Total dynamic write energy per access \(nJ\): (?P<energy>[0-9.]+)"),
        }

    def __call__(self, port, size, readOrWrite):
        with open(self.cacti["cfgInputFile"], "w") as fout:
            fout.write(self.cfgTemplate.format(int(size), int(port)))

        try:
            p = subprocess.run(
                f"cd {self.cacti['path']};./cacti -infile {self.cacti['cfgInputFile']}", shell=True, capture_output=True)
        except Exception as e:
            print(e)
            return 0, 0

        rst = str(p.stdout, encoding="utf-8")
        if p.returncode != 0:
            print(p)
            print("CACTI failed!")
            exit(1)

        areaMatch = re.search(self.pattern["area"], rst)
        if areaMatch is None:
            area = 0
            print("Can't measure area!")
        else:
            area = float(areaMatch.group("width")) * \
                float(areaMatch.group("height"))

        cycle = float(re.search(self.pattern["cycle"], rst).group("cycle"))

        if readOrWrite == WRProperty.READ:
            energy = float(
                re.search(self.pattern["readE"], rst).group("energy"))
        elif readOrWrite == WRProperty.WRITE:
            energy = float(
                re.search(self.pattern["writeE"], rst).group("energy"))
        else:
            energy = float(re.search(self.pattern["readE"], rst).group("energy")) +\
                float(re.search(self.pattern["writeE"], rst).group("energy"))

        power = energy / cycle

        return area * self.cacti["areaUnit"], power * self.cacti["powerUnit"]
