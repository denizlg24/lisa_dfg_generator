import random
import os
from random import randint
from graph_gen import *
from dfg_gnn_attributes_generator import *
from dfg import DFGGraph, Vertex
from tqdm import tqdm
import argparse

enable_cgra_me = True
enable_morpher = False


def myHandler(signum, frame):
    raise Exception("TimeoutError")


def dump_cgra_me_graph(dir, graph: DFGGraph):
    graph_name = graph.name

    with open(os.path.join(dir, "cgra_me", graph_name + ".dot"), "w") as f:
        f.write("digraph G { \n")
        f.write(graph.cgrame_toStr())
        f.write("}\n")

    return True


def dump_morpher_graph(dir, graph: DFGGraph):
    graph_name = graph.name

    with open(os.path.join(dir, "morpher", graph_name + ".xml"), "w") as f:
        f.write(graph.morpher_toStr())

    return True


def single_dfg_gen(dir, i):
    """
    i: the id of graph
    """
    MIN_NODE = 40
    MAX_NODE = 100

    number_node = random.choice(range(MIN_NODE, MAX_NODE))
    # print("i", "number of node", number_node)
    min_edge = 2  # for each node
    max_edge = randint(3, 4)  # for each node
    edge_dic = dfg_json_maker(str(1), 0, 0, number_node, min_edge, max_edge, 0, 1, 2, 1)

    graph = DFGGraph(str(i))
    for num in range(number_node):
        graph.add_vertex(Vertex(id=str(num + 1)))

    for key, values in edge_dic.items():
        for value in values:
            graph.add_edge(key, value)

    node_number = len(graph.vertices.keys())
    graph.check_connectivity()
    graph.handle_cycle()
    if len(graph.vertices) == 0:
        return False
    if enable_cgra_me:
        if not graph.satisfy_cgra_me_constraint():
            # print("did not generate", i)
            return False
    elif enable_morpher:
        if not graph.check_morpher_edge_limit():
            # print("did not generate", i)
            return False
    else:
        assert False

    if not graph.check_connectivity():
        #
        return False
    new_node_number = len(graph.vertices.keys())
    # if node_number != new_node_number:
    # add somework to handle it
    graph.make_node_index_continous(node_number)

    if enable_morpher:
        graph.assign_morpher_op_code()
    elif enable_cgra_me:
        graph.assign_cgra_me_op_code()

    # try:
    #     signal.signal(signal.SIGALRM, myHandler)
    #     signal.alarm(10)
    #
    #     asap_value = graph.ASAP()
    #
    #     signal.alarm(0)
    # except Exception as ret:
    #     print("msg:", ret)
    #     graph.ASAP()
    #     with open(os.path.join(dir, "graph", "error.txt"), "w") as f:
    #         for edge in graph.edges:
    #             start_node, end_node = edge
    #             f.write(str(start_node - 1) + '\t' + str(end_node - 1) + '\n')
    asap_value = graph.set_ASAP()
    graph.set_node_feature()
    graph.get_same_level_node()

    if len(graph.vertices) == 0:
        return False

    # save graph info
    # Because graph in torch geometric counts vertices from 0, all generated nodes id will -1.
    with open(os.path.join(dir, "graph", str(i) + ".txt"), "w") as f:
        for edge in graph.edges:
            start_node, end_node = edge
            f.write(str(start_node) + "\t" + str(end_node) + "\n")

    if enable_cgra_me:
        dump_cgra_me_graph(dir, graph)
    elif enable_morpher:
        dump_morpher_graph(dir, graph)

    # save tag info
    with open(os.path.join(dir, "graph", str(i) + "_feature.txt"), "w") as f:
        if asap_value is None:
            return
        for idx in range(len(asap_value)):
            f.write(str(asap_value[idx]) + "\n")

    with open(os.path.join(dir, "graph", str(i) + "_op.txt"), "w") as f:
        for idx in range(len(asap_value)):
            f.write(str(graph.vertices[idx].opcode) + "\n")

    return True


def generator(n_data, dir, model, satrt_index=0):
    """
    n_data: long, number of graphs as training data set
    """

    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    if not os.path.exists(os.path.join(dir, "graph")):
        os.mkdir(os.path.join(dir, "graph"))
    if not os.path.exists(os.path.join(dir, model)):
        os.mkdir(os.path.join(dir, model))
    # if not os.path.exists(os.path.join(dir, "label")):
    #     os.mkdir(os.path.join(dir, "label"))

    total_attempts = 0
    for i in tqdm(range(satrt_index, n_data + satrt_index)):
        attempts = 0
        while True:
            attempts += 1
            total_attempts += 1
            if single_dfg_gen(dir, i):
                break
        if attempts > 1:
            tqdm.write(f"graph {i}: {attempts} attempts")
    avg = total_attempts / max(n_data, 1)
    print(f"generation done: {total_attempts} total attempts for {n_data} graphs (avg {avg:.1f} per graph)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dfg_generator parameter.")
    parser.add_argument(
        "-s", "--start_index", default=0, type=int, help="the start index of graph"
    )
    parser.add_argument(
        "-n",
        "--graph_num",
        default=100,
        type=int,
        help="the number of generated graphes",
    )
    parser.add_argument(
        "-d",
        "--directory",
        default="datasets",
        help="the directory of generated graphes",
    )

    args = parser.parse_args()
    # print(args.accumulate(args.integers))
    print("number of graph:", args.graph_num)
    print("start index", args.start_index)
    print("directory", args.directory)

    # assert(False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    data_dir = os.path.join(repo_root, "data", args.directory)

    model = "morpher" if enable_morpher else "cgra_me"

    generator(
        args.graph_num,
        data_dir,
        model=model,
        satrt_index=args.start_index,
    )

    transform_graph_by_dir(data_dir, "graph", "graph")
