import sys
import re
import numpy as np
import json
import os
from os import listdir
import matplotlib.pyplot as plt

def draw_performance_graphs_cpu(graphdata, title, algorithm, name_filter=""):
    labels = list()
    openmp_data = list()
    openmp_std = list()
    openacc_data = list()
    openacc_std = list()

    to_process = list()

    if name_filter is not "":
        for graph in graphdata:
            if name_filter in graph:
                to_process.append(graph)

    for graph in to_process:
        labels.append(graph)
        openmp_data.append(graphdata[graph][algorithm + "_OPENMP"]["avg_total"])
        openmp_std.append(graphdata[graph][algorithm + "_OPENMP"]["std_total"])
        openacc_data.append(graphdata[graph][algorithm + "_OPENACC"]["avg_total"])
        openacc_std.append(graphdata[graph][algorithm + "_OPENACC"]["std_total"])

    # Normalise
    for i in range(len(openmp_data)):
        max_val = max(openmp_data[i], openacc_data[i])
        openmp_data[i] /= max_val
        openmp_std[i] /= max_val
        openacc_data[i] /= max_val
        openacc_std[i] /= max_val

    # Draw plots
    ind = np.arange(len(openmp_data))
    width = 0.35

    fig, ax = plt.subplots()
    ax.tick_params(axis='x', rotation=-25)
    ax.tick_params(axis='both', labelsize='large')
    rects1 = ax.bar(ind - width/2, openmp_data, width, yerr=cuda_std, label='OpenMP', color="#76b900")
    rects2 = ax.bar(ind + width/2, openacc_data, width, yerr=openacc_std, label='OpenACC', color="#288ec1")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalised runtime')
    ax.set_xlabel('Graph name')
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_yticks(np.linspace(0, 1, 21))
    ax.set_xticklabels([x.replace(name_filter + "-", "") for x in labels])
    ax.legend()
    plt.tight_layout()

def draw_performance_graphs(graphdata, title, algorithm, name_filter=""):
    labels = list()
    cuda_data = list()
    cuda_std = list()
    openacc_data = list()
    openacc_std = list()

    to_process = list()

    if name_filter is not "":
        for graph in graphdata:
            if name_filter in graph:
                to_process.append(graph)

    for graph in to_process:
        labels.append(graph)
        cuda_data.append(graphdata[graph][algorithm + "_CUDA"]["avg_total"])
        cuda_std.append(graphdata[graph][algorithm + "_CUDA"]["std_total"])
        openacc_data.append(graphdata[graph][algorithm + "_OPENACC"]["avg_total"])
        openacc_std.append(graphdata[graph][algorithm + "_OPENACC"]["std_total"])

    # Normalise
    for i in range(len(cuda_data)):
        max_val = max(cuda_data[i], openacc_data[i])
        cuda_data[i] /= max_val
        cuda_std[i] /= max_val
        openacc_data[i] /= max_val
        openacc_std[i] /= max_val

    # Draw plots
    ind = np.arange(len(cuda_data))
    width = 0.35

    fig, ax = plt.subplots()
    ax.tick_params(axis='x', rotation=-25)
    ax.tick_params(axis='both', labelsize='large')
    rects1 = ax.bar(ind - width/2, cuda_data, width, yerr=cuda_std, label='CUDA', color="#76b900")
    rects2 = ax.bar(ind + width/2, openacc_data, width, yerr=openacc_std, label='OpenACC', color="#288ec1")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Normalised runtime')
    ax.set_xlabel('Graph name')
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_yticks(np.linspace(0, 1, 21))
    ax.set_xticklabels([x.replace(name_filter + "-", "") for x in labels])
    ax.legend()
    plt.tight_layout()

def draw_all_graphs(data):
    # Create Pagerank performance graphs
    draw_performance_graphs(data, "Normalised pagerank runtimes of graph500 graphs", "PAGERANK", "graph500")
    draw_performance_graphs(data, "Normalised pagerank runtimes of KONECT graphs", "PAGERANK", "KONECT")
    draw_performance_graphs(data, "Normalised pagerank runtimes of SNAP graphs", "PAGERANK", "SNAP")

    # Create BFS performance graphs

    draw_performance_graphs(data, "Normalised BFS runtimes of graph500 graphs", "BFS", "graph500")
    draw_performance_graphs(data, "Normalised BFS runtimes of KONECT graphs", "BFS", "KONECT")
    draw_performance_graphs(data, "Normalised BFS runtimes of SNAP graphs", "BFS", "SNAP")

    plt.show()

if __name__ == "__main__":
    results = {}

    data_regex = re.compile(r"[0-9]+.[0-9]+\ [0-9]+.[0-9]+\ [0-9]+.[0-9]+")

    directory = sys.argv[1]

    for file in os.listdir(directory):
        if "stderr" not in file:
            full_filename = directory + file
            current_section = ""
            current_graph = ""
            with open(full_filename) as fp:
                for line in fp:
                    line = line[:-1]
                    if line.startswith("SECTION:"):
                        current_section = line[8:]
                        if current_section not in results.keys():
                            results[current_section] = {}
                    elif line.startswith("GRAPH:"):
                        current_graph = line[6:]
                        if current_graph not in results[current_section].keys():
                            results[current_section][current_graph] = {"total": list(), "computation": list(), "memory": list()}
                    elif data_regex.match(line):
                        split_line = line.split()
                        results[current_section][current_graph]["total"].append(float(split_line[0]))
                        results[current_section][current_graph]["computation"].append(float(split_line[1]))
                        results[current_section][current_graph]["memory"].append(float(split_line[2]))
                    elif line.startswith("#"):
                        pass
                    else:
                        # Invalid data
                        print("Invalid data in section " + current_section +" and graph " + current_graph)

    # Data processing
    for section in results:
        for graph in results[section]:
            if(len(results[section][graph]["total"]) > 1):
                # Remove outliers due to caching
                results[section][graph]["total"] = results[section][graph]["total"][1:]
                results[section][graph]["computation"] = results[section][graph]["computation"][1:]
                results[section][graph]["memory"] = results[section][graph]["memory"][1:]
                # Convert to numpy
                results[section][graph]["total"] = np.array(results[section][graph]["total"])
                results[section][graph]["computation"] = np.array(results[section][graph]["computation"])
                results[section][graph]["memory"] = np.array(results[section][graph]["memory"])

                # Calculate statistics
                results[section][graph]["avg_total"] = np.mean(results[section][graph]["total"])
                results[section][graph]["std_total"] = np.std(results[section][graph]["total"])
                results[section][graph]["avg_computation"] = np.mean(results[section][graph]["computation"])
                results[section][graph]["std_computation"] = np.std(results[section][graph]["computation"])
                results[section][graph]["avg_memory"] = np.mean(results[section][graph]["memory"])
                results[section][graph]["std_memory"] = np.std(results[section][graph]["memory"])

    # print(results)

    # Format the data to graphs first
    graphdata = {}

    # Find all graphs
    for section in results:
        for graph in results[section]:
            if graph not in graphdata.keys():
                graphdata[graph] = {}

    # Fill in the data
    for section in results:
        for graph in results[section]:
            graphdata[graph][section] = {"avg_total": -1.0, "std_total": -1.0,
                                        "avg_computation": -1.0, "std_computation": -1.0,
                                        "avg_memory": -1.0, "std_memory": -1.0}

            for data in graphdata[graph][section].keys():
                if data in results[section][graph].keys():
                    graphdata[graph][section][data] = results[section][graph][data]

    sorted_graphdata = {}
    for key in sorted(graphdata):
        sorted_graphdata[key] = graphdata[key]

    graphdata = sorted_graphdata

    print(json.dumps(graphdata, indent=4))

    draw_all_graphs(graphdata)
