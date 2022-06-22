import sys
import re
import numpy as np
import json
import os
from os import listdir
import matplotlib.pyplot as plt

def print_data_line(graphname, col1, col2, col3, col4, col5, col6):
    print("\multicolumn{1}{|l|}{" + str(graphname) + "}      & \multicolumn{1}{l|}{" + str(col1) + "}      & \multicolumn{1}{l|}{" + str(col2) + "}            &    " + str(col3) + "   & \multicolumn{1}{|l|}{" + str(col4) + "}      & \multicolumn{1}{l|}{" + str(col5) + "}            &  \multicolumn{1}{l|}{" + str(col6) + "}      \\\ \hline")

def print_table(graphdata, alg, graphfilter=""):
    print("""
\\begin{table}[H]
\\begin{tabular}{lllllll}
\hline
\multicolumn{1}{|l|}{Graph}  &                            & OpenMP                             & \multicolumn{1}{l|}{}       &                            & OpenACC                          & \multicolumn{1}{l|}{}       \\\ \hline
\multicolumn{1}{l|}{}        & \multicolumn{1}{l|}{Total} & \multicolumn{1}{l|}{Main loop} & \multicolumn{1}{l|}{Memory} & \multicolumn{1}{l|}{Total} & \multicolumn{1}{l|}{Main loop} & \multicolumn{1}{l|}{Memory} \\\ \hline"""
        )

    toprocess = list()
    for graph in graphdata:
        if graphfilter is "":
            toprocess.append(graph)
        else:
            if graphfilter in graph:
                toprocess.append(graph)
            else:
                pass

    roundto = 5
    for graph in toprocess:
        cuda_total = "\\begin{tabular}[c]{@{}l@{}}" + str(round(graphdata[graph][alg.upper() + "_OPENMP"]["avg_total"], roundto)) + "\\\ $ \\pm  $ " + str(round(graphdata[graph][alg.upper() + "_OPENMP"]["std_total"], roundto)) + "\end{tabular}"
        cuda_comp = "\\begin{tabular}[c]{@{}l@{}}" + str(round(graphdata[graph][alg.upper() + "_OPENMP"]["avg_computation"], roundto)) + "\\\ $ \\pm  $ " + str(round(graphdata[graph][alg.upper() + "_OPENMP"]["std_computation"], roundto)) + "\end{tabular}"
        cuda_mem = "\\begin{tabular}[c]{@{}l@{}}" + str(round(graphdata[graph][alg.upper() + "_OPENMP"]["avg_memory"], roundto)) + "\\\ $ \\pm  $ " + str(round(graphdata[graph][alg.upper() + "_OPENMP"]["std_memory"], roundto)) + "\end{tabular}"
        acc_total = "\\begin{tabular}[c]{@{}l@{}}" + str(round(graphdata[graph][alg.upper() + "_OPENACC"]["avg_total"], roundto)) + "\\\ $ \\pm  $ " + str(round(graphdata[graph][alg.upper() + "_OPENACC"]["std_total"], roundto)) + "\end{tabular}"
        acc_comp = "\\begin{tabular}[c]{@{}l@{}}" + str(round(graphdata[graph][alg.upper() + "_OPENACC"]["avg_computation"], roundto)) + "\\\ $ \\pm  $ " + str(round(graphdata[graph][alg.upper() + "_OPENACC"]["std_computation"], roundto)) + "\end{tabular}"
        acc_mem = "\\begin{tabular}[c]{@{}l@{}}" + str(round(graphdata[graph][alg.upper() + "_OPENACC"]["avg_memory"], roundto)) + "\\\ $ \\pm  $ " + str(round(graphdata[graph][alg.upper() + "_OPENACC"]["std_memory"], roundto)) + "\end{tabular}"

        print_data_line(graph.replace(graphfilter, "").replace("_", "\_"), cuda_total, cuda_comp, cuda_mem, acc_total, acc_comp, acc_mem)

    # Footer prints
    print("\end{tabular}")
    print("\caption{}")
    print("\label{tab:" + alg.lower() + "_" + graphfilter[:-1].lower() + "_raw}")
    print("\end{table}\n\n")

if __name__ == "__main__":

    graphdata = {}

    with open(sys.argv[1]) as fp:
        content = fp
        graphdata = json.load(content)

    sorted_graphdata = {}
    for key in sorted(graphdata):
        sorted_graphdata[key] = graphdata[key]

    graphdata = sorted_graphdata

    print("\subsection{graph500}")
    print_table(graphdata, "PAGERANK", "graph500-")
    print("\subsection{KONECT}")
    print_table(graphdata, "PAGERANK", "KONECT-")
    print("\subsection{SNAP}")
    print_table(graphdata, "PAGERANK", "SNAP-")
