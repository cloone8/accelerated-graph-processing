import sys
import re
import numpy as np
import json
import os
from os import listdir
import matplotlib.pyplot as plt
from process_from_raw import *

if __name__ == "__main__":

    graphdata = {}

    with open(sys.argv[1]) as fp:
        content = fp
        graphdata = json.load(content)

    sorted_graphdata = {}
    for key in sorted(graphdata):
        sorted_graphdata[key] = graphdata[key]

    graphdata = sorted_graphdata

    draw_all_graphs(graphdata)
