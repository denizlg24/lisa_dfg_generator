import sys

import random
import xml.etree.cElementTree as ET

from math import ceil
from graph_gen import *
from random import randint


min_asap = 1
min_alap = 1

morpher_op_code = dict()
morpher_op_code[0] = ["OLOAD", "MOVC"]
morpher_op_code[1] = ["LOADB", "CMERGE", "CLT", "OR", "OSTORE", "LS", "LOAD"]
morpher_op_code[2] = ["SELECT", "ADD", "MUL", "STOREB"]
morpher_op_code[3] = ["SELECT", "STORE"]


morpher_nonstore_opcode = dict()
morpher_nonstore_opcode[0] = ["OLOAD", "MOVC"]
morpher_nonstore_opcode[1] = ["LOADB", "CMERGE", "CLT", "OR", "LS", "LOAD"]
morpher_nonstore_opcode[2] = ["SELECT", "ADD", "MUL"]
morpher_nonstore_opcode[3] = ["SELECT"]

morpher_store_opcode = dict()
morpher_store_opcode[1] = ["OSTORE"]
morpher_store_opcode[2] = ["STOREB"]
morpher_store_opcode[3] = ["STORE"]


# morpher_op_code["OLOAD"] = 0
# morpher_op_code["MOVC"] = 0
# morpher_op_code["LOADB"] = 1
# morpher_op_code["CMERGE"] = 1
# morpher_op_code["CLT"] = 1
# morpher_op_code["OR"] = 1
# morpher_op_code["OSTORE"] = 1
# morpher_op_code["LS"] = 1
# morpher_op_code["LOAD"] = 1
# morpher_op_code["SELECT"] = 2
# morpher_op_code["ADD"] = 2
# morpher_op_code["MUL"] = 2
# morpher_op_code["STOREB"] = 2
# morpher_op_code["SELECT"] = 3


def indent(elem, level=0):
    i = "\n"

    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + ""
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
            elem.tail = i
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


class Vertex:
    """Represent a vertex with a label and possible connected component."""

    # pylint: disable=too-few-public-methods
    # Using class so it's hashable, even though it doesn't have public methods
    def __init__(self, id, opcode=""):
        self.id = id
        self.opcode = opcode
        self.asap = 0
        self.in_degree = 0
        self.out_degree = 0
        self.no_grandparent = 0
        self.no_grandchild = 0
        self.no_ancestor = 0
        self.no_descendant = 0
        self.is_mem = 0
        self.output_dest_port = dict()

    def feature_str(self):
        # return (str(int(self.asap) * 2)+" "+str(self.in_degree) + "" +str(self.out_degree)+" "+str(self.no_grandparent) + "" +str(self.no_grandchild))
        return (
            str(self.asap)
            + " "
            + str(self.in_degree)
            + " "
            + str(self.out_degree)
            + " "
            + str(self.no_grandparent)
            + " "
            + str(self.no_grandchild)
            + " "
            + str(self.no_ancestor)
            + " "
            + str(self.no_descendant)
            + " "
            + str(self.is_mem)
        )

    def __repr__(self):
        return "Vertex: " + self.id + self.opcode


class DFGGraph:
    """Represent a graph as a dictionary of vertices mapping labels to edges."""

    def __init__(self, name_):
        self.name = name_
        self.vertices = {}
        self.num_vertices = 0
        self.edges = set()
        self.backedges = set()
        self.pred = {}
        self.succ = {}
        self.asap = {}
        self.alap = {}

    def add_vertex(self, vertex):
        """Add a new vertex, optionally with edges to other vertices."""
        if vertex in self.vertices:
            raise Exception("Error: adding vertex that already exists")

        self.vertices[int(vertex.id)] = vertex
        self.num_vertices += 1
        self.pred[int(vertex.id)] = set()  # this means node parent
        self.succ[int(vertex.id)] = set()  # this means node children

    def add_edge(self, start, end, bidirectional=False):
        """Add a edge (default bidirectional) between two vertices."""
        if start not in self.vertices.keys() or end not in self.vertices.keys():
            raise Exception("Vertices to connect not in graph!", start, end)
        self.edges.add((start, end))
        self.pred[end].add(start)
        self.succ[start].add(end)
        if bidirectional:
            self.edges.add((end, start))
            self.pred[start].add(end)
            self.succ[end].add(start)

    def DFS_cycle_util(self, v, visited, trace_stack, cycle_edge):

        # Mark the current node as visited
        # and print it
        visited.add(v)
        trace_stack.add(v)
        # print(v, end=' ')

        # Recur for all the vertices
        # adjacent to this vertex
        for succ in self.succ[v]:
            if succ in trace_stack:
                # get cycle edge
                cycle_edge.add((v, succ))
            else:
                new_stack = trace_stack.copy()
                self.DFS_cycle_util(succ, visited, new_stack, cycle_edge)

    def check_connectivity(self):
        temp_vert = self.vertices.copy()
        self.vertices.clear()
        for node in temp_vert.keys():
            # remove lonely node
            if len(self.pred[node]) != 0 or len(self.succ[node]) != 0:
                # print(node, len(self.pred[node]), len(self.succ[node]))
                self.vertices[node] = temp_vert[node]

        # print()

        start_node = set()
        for node in self.vertices.keys():
            if len(self.pred[node]) == 0:
                start_node.add(node)
        if len(start_node) == 0:
            return False
        else:
            return True

    def make_node_index_continous(self, max_node_index):
        # print("vertex:", self.vertices)
        # print("edges:",self.edges)
        # print("backtrack_edges:",self.backtrack_edges)
        # print("pred", self.pred)
        # print("succ", self.succ)

        node_index = 0
        old_to_new = {}
        for i in range(max_node_index + 1):
            if i in self.vertices.keys():
                old_to_new[i] = node_index
                node_index += 1

        assert node_index == len(self.vertices.keys())

        # print("old_to_new:", old_to_new)

        temp_vertex = self.vertices.copy()
        self.vertices.clear()
        self.pred.clear()
        self.succ.clear()
        for old_id, new_id in old_to_new.items():
            self.vertices[new_id] = Vertex(id=str(new_id))
            self.pred[new_id] = set()
            self.succ[new_id] = set()

        temp_edges = self.edges.copy()
        self.edges.clear()

        for src, des in temp_edges:
            new_src = old_to_new[src]
            new_des = old_to_new[des]
            self.edges.add((new_src, new_des))
            self.pred[new_des].add(new_src)
            self.succ[new_src].add(new_des)

        temp_back_edges = self.backedges.copy()
        self.backedges.clear()
        for src, des in temp_back_edges:
            if (src not in old_to_new.keys()) or (des not in old_to_new.keys()):
                continue
            new_src = old_to_new[src]
            new_des = old_to_new[des]
            self.backedges.add((new_src, new_des))

        # print("vertex:", self.vertices)
        # print("edges:",self.edges)
        # print("backtrack_edges:",self.backtrack_edges)
        # print("pred", self.pred)
        # print("succ", self.succ)

    def handle_cycle(self):

        self.backedges.clear()

        while True:
            cycle_edges = set()
            visited = set()
            start_node = set()
            for node in self.vertices.keys():
                if len(self.pred[node]) == 0:
                    start_node.add(node)

            # print("start_node", start_node)

            for node in start_node:
                trace_stack = set()
                self.DFS_cycle_util(node, visited, trace_stack, cycle_edges)

            # print("cycle edge", cycle_edges)
            # reconstruct the node list and pred  & succ dic
            temp_vert = self.vertices.copy()
            self.vertices.clear()
            self.pred.clear()
            self.succ.clear()
            for node in temp_vert.keys():
                if node in visited:
                    self.vertices[node] = temp_vert[node]
                    self.succ[node] = set()
                    self.pred[node] = set()

            num_old_back_edge = len(self.backedges)
            for edge in cycle_edges:
                self.backedges.add((edge[0], edge[1]))
            temp_edges = self.edges.copy()
            self.edges.clear()

            # remove cycle_edge
            for edge in temp_edges:
                start = edge[0]
                end = edge[1]
                is_backtrack_edge = False
                for b_edge in cycle_edges:
                    if b_edge[0] == start and b_edge[1] == end:
                        is_backtrack_edge = True
                        break
                if not is_backtrack_edge:
                    if edge[0] in visited and edge[1] in visited:
                        self.edges.add(edge)
                        self.pred[edge[1]].add(edge[0])
                        self.succ[edge[0]].add(edge[1])
            # print("vertices ", self.vertices)
            # print("pred ", self.pred)
            # print("succ ", self.succ)
            temp_node = self.vertices.copy()
            self.vertices.clear()
            for node in temp_node.keys():
                if len(self.pred[node]) == 0 and len(self.succ[node]) == 0:
                    continue
                else:
                    self.vertices[node] = temp_node[node]

            # print("after process vertices ", self.vertices)
            if len(cycle_edges) == 0:
                break

    def set_ASAP(self):
        # check Predecessor
        asap_value = {}
        non_scheduled = set()

        temp_pred = self.pred.copy()
        for node in self.vertices:
            non_scheduled.add(node)

        # print("edges:", self.edges)
        # print("edge precedence:", temp_pred)

        temp = set()
        for node in non_scheduled:
            if len(temp_pred[node]) == 0:
                asap_value[node] = min_asap
            else:
                temp.add(node)

        if len(temp) == len(non_scheduled):
            Exception("could not schedule")
            return
        non_scheduled = temp.copy()

        tried = 0
        while len(non_scheduled) > 0:
            for node in non_scheduled:
                pred_finished = True
                for p in temp_pred[node]:
                    if p in non_scheduled:
                        pred_finished = False
                        break
                if pred_finished:
                    max_asap = 1
                    for p in temp_pred[node]:
                        if asap_value[p] > max_asap:
                            max_asap = asap_value[p]
                    asap_value[node] = max_asap + 1
                    non_scheduled.remove(node)
                    break
            tried += 1
            if tried == 10000:
                print(non_scheduled)
                print(self.edges)
                print(self.pred)
                Exception("should not happen")

        # print("ASAP value: ",asap_value)
        for key, value in asap_value.items():
            self.vertices[key].asap = value

        self.asap = asap_value
        return asap_value

    def generate_simple_labels(self, asap_value, indegree_threashold):
        # if the in-degree > indegree_threashold, the label =  asap_value - 1.
        # Label value must >= 0
        nodeL_labels = {}
        for node, avalue in asap_value.items():
            value = avalue
            in_degree = len(self.pred[node])
            if in_degree == 0:
                value = avalue
            elif in_degree >= indegree_threashold:
                value -= int(in_degree / indegree_threashold)
            else:
                value += indegree_threashold - in_degree
            if value < 0:
                value = 0
            nodeL_labels[node] = value
        # print("node labels:", nodeL_labels)
        return nodeL_labels

    def set_in_degree(self):
        in_degree = {}
        for node in self.vertices.keys():
            node_in_degree = len(self.pred[node])
            in_degree[node] = node_in_degree
            self.vertices[node].in_degree = node_in_degree
        return in_degree

    def set_out_degree(self):
        out_degree = {}
        for node in self.vertices.keys():
            node_out_degree = len(self.succ[node])
            out_degree[node] = node_out_degree
            self.vertices[node].out_degree = node_out_degree
        return out_degree

    def set_no_grandparent(self):
        for node, vert in self.vertices.items():
            grandparent = set()
            for pred in self.pred[node]:
                for gpred in self.pred[pred]:
                    grandparent.add(gpred)
            vert.no_grandparent = len(grandparent)

    def set_no_grandchild(self):
        for node, vert in self.vertices.items():
            grandchild = set()
            for succ in self.succ[node]:
                for gsucc in self.succ[succ]:
                    grandchild.add(gsucc)
            vert.no_grandchild = len(grandchild)

    def set_no_ancestor(self):
        ancestors = {}
        for node in self.vertices.keys():
            ancestors[node] = set()

        max_asap = 0
        for node, asap in self.asap.items():
            max_asap = max(max_asap, asap)

        for asap_val in range(0, max_asap + 1):
            for node, node_asap in self.asap.items():
                if node_asap == asap_val:
                    for pred in self.pred[node]:
                        ancestors[node].add(pred)
                        for anc in ancestors[pred]:
                            ancestors[node].add(anc)

        for node in ancestors.keys():
            self.vertices[node].no_ancestor = len(ancestors[node])

    def set_no_descendant(self):
        descendants = {}
        for node in self.vertices.keys():
            descendants[node] = set()

        max_asap = 0
        for node, asap in self.asap.items():
            max_asap = max(max_asap, asap)

        for asap_val in range(max_asap, -1, -1):
            for node, node_asap in self.asap.items():
                if node_asap == asap_val:
                    for succ in self.succ[node]:
                        descendants[node].add(succ)
                        for desc in descendants[succ]:
                            descendants[node].add(desc)

        for node in descendants.keys():
            self.vertices[node].no_descendant = len(descendants[node])
        return

    def number_dependent_nodes_in_between(self, ancestor, descendant):

        max_asap = 0
        for node, node_asap in self.asap.items():
            max_asap = max(max_asap, node_asap)

        descendants = {}
        ancestors = {}
        for node in self.vertices.keys():
            descendants[node] = set()
            ancestors[node] = set()

        for asap_val in range(max_asap, -1, -1):
            for node, node_asap in self.asap.items():
                if node_asap == asap_val:
                    for succ in self.succ[node]:
                        descendants[node].add(succ)
                        for desc in descendants[succ]:
                            descendants[node].add(desc)

        for asap_val in range(0, max_asap + 1):
            for node, node_asap in self.asap.items():
                if node_asap == asap_val:
                    for pred in self.pred[node]:
                        ancestors[node].add(pred)
                        for anc in ancestors[pred]:
                            ancestors[node].add(anc)

        ancestor_asap = self.asap[ancestor]
        descendant_asap = self.asap[descendant]

        num_dependent_node = 0
        for node, node_asap in self.asap.items():
            if node in descendants[ancestor] and node in ancestors[descendant]:
                num_dependent_node += 1
        return num_dependent_node

    def number_nodes_in_between(self, a_asap, b_asap):
        num_inbetween_node = 0
        for node, node_asap in self.asap.items():
            if a_asap < node_asap < b_asap or b_asap < node_asap < a_asap:
                num_inbetween_node += 1
        return num_inbetween_node

    def get_same_asap_nodes(self, asap_value):
        same_asap_node = 0
        for node, node_asap in self.asap.items():
            if node_asap == asap_value:
                same_asap_node += 1
        return same_asap_node

    def generate_edge_feature(self):
        final_str = ""
        for src, des in self.edges:
            src_asap = self.asap[src]
            des_asap = self.asap[des]
            ancestor = self.vertices[src].no_ancestor
            descendant = self.vertices[src].no_descendant
            num_inbetween_node = 0
            same_asap_node = 0
            for node, node_asap in self.asap.items():
                if src_asap < node_asap < des_asap:
                    num_inbetween_node += 1
                if node_asap == src_asap or node_asap == des_asap:
                    same_asap_node += 1
            final_str += (
                str(src)
                + " "
                + str(des)
                + " "
                + str(num_inbetween_node)
                + " "
                + str(same_asap_node)
                + " "
                + str(des_asap - src_asap)
                + " "
                + str(ancestor)
                + " "
                + str(descendant)
                + "\n"
            )
        return final_str

    def get_same_level_node(self):
        max_asap = 0
        for node, node_asap in self.asap.items():
            max_asap = max(max_asap, node_asap)

        descendants = {}
        ancestors = {}
        for node in self.vertices.keys():
            descendants[node] = set()
            ancestors[node] = set()

        for asap_val in range(max_asap, -1, -1):
            for node, node_asap in self.asap.items():
                if node_asap == asap_val:
                    for succ in self.succ[node]:
                        descendants[node].add(succ)
                        for desc in descendants[succ]:
                            descendants[node].add(desc)

        for asap_val in range(0, max_asap + 1):
            for node, node_asap in self.asap.items():
                if node_asap == asap_val:
                    for pred in self.pred[node]:
                        ancestors[node].add(pred)
                        for anc in ancestors[pred]:
                            ancestors[node].add(anc)
        final_str = ""

        for i in range(min_asap, max_asap + 1):
            same_level_node = []
            for node, node_asap in self.asap.items():
                if node_asap == i:
                    same_level_node.append(node)

            same_level_node_dist = {}

            for a_node in same_level_node:
                for b_node in same_level_node:
                    if a_node <= b_node:
                        continue
                    # print(a_node, b_node)
                    common_des = []
                    for des in descendants[a_node]:
                        if des in descendants[b_node]:
                            common_des.append((des, self.asap[des]))

                    common_anc = []
                    for des in ancestors[a_node]:
                        if des in ancestors[b_node]:
                            common_anc.append((des, self.asap[des]))
                    level_asap = self.asap[a_node]
                    exist_des_and_anc = 0
                    if len(common_des) > 0 or len(common_anc) > 0:
                        if len(common_des) > 0 and len(common_anc) > 0:
                            exist_des_and_anc = 1
                        common_des.sort(key=lambda x: x[1])
                        common_anc.sort(key=lambda x: x[1])
                        # print(common_des)
                        ancestor_dist = 0
                        descendant_dist = 0
                        ancestor_node_ine_betweeen = 0
                        descendant_node_ine_betweeen = 0
                        num_parallel_node = self.get_same_asap_nodes(level_asap)
                        dependent_node_with_ancestor = 0
                        dependent_node_with_descendant = 0

                        if len(common_anc) > 0:
                            ancestor_dist = level_asap - common_anc[0][1]
                            num_parallel_node += self.get_same_asap_nodes(
                                common_anc[0][1]
                            )
                            ancestor_node_ine_betweeen = self.number_nodes_in_between(
                                common_anc[0][1], level_asap
                            )
                            dependent_node_with_ancestor = (
                                self.number_dependent_nodes_in_between(
                                    common_anc[0][0], a_node
                                )
                            )
                            dependent_node_with_ancestor += (
                                self.number_dependent_nodes_in_between(
                                    common_anc[0][0], b_node
                                )
                            )
                        if len(common_des) > 0:
                            num_parallel_node += self.get_same_asap_nodes(
                                common_des[-1][1]
                            )
                            descendant_dist = (
                                common_des[len(common_des) - 1][1] - level_asap
                            )
                            descendant_node_ine_betweeen = self.number_nodes_in_between(
                                common_des[-1][1], level_asap
                            )
                            dependent_node_with_descendant += (
                                self.number_dependent_nodes_in_between(
                                    a_node, common_des[-1][0]
                                )
                            )
                            dependent_node_with_descendant += (
                                self.number_dependent_nodes_in_between(
                                    b_node, common_des[-1][0]
                                )
                            )

                        same_level_node_dist[(a_node, b_node)] = [
                            ancestor_dist,
                            descendant_dist,
                            ancestor_node_ine_betweeen,
                            descendant_node_ine_betweeen,
                            num_parallel_node,
                            dependent_node_with_ancestor,
                            dependent_node_with_descendant,
                            exist_des_and_anc,
                        ]
                        # same_level_node_dist[(a_node, b_node)] = [descendant_dist, \
                        #   descendant_node_ine_betweeen, num_parallel_node,  dependent_node_with_descendant]
            # shortest path
            # number of nodes to the common consumer or producer node
            for nodes, dist in same_level_node_dist.items():
                final_str += str(nodes[0]) + " " + str(nodes[1])
                for item in dist:
                    final_str += " " + str(item)
                final_str += "\n"

        return final_str

    def get_start_node(self):
        start_nodes = set()
        for node, node_asap in self.asap.items():
            if node_asap == min_asap:
                start_nodes.add(node)

        descendants = {}
        for node in self.vertices.keys():
            descendants[node] = set()

        max_asap = 0
        for node, asap in self.asap.items():
            max_asap = max(max_asap, asap)

        for asap_val in range(max_asap, -1, -1):
            for node, node_asap in self.asap.items():
                if node_asap == asap_val:
                    for succ in self.succ[node]:
                        descendants[node].add(succ)
                        for desc in descendants[succ]:
                            descendants[node].add(desc)

        start_node_dist = {}

        for a_node in start_nodes:
            for b_node in start_nodes:
                if a_node >= b_node:
                    continue
                # print(a_node, b_node)
                common_des = []
                for des in descendants[a_node]:
                    if des in descendants[b_node]:
                        common_des.append((des, self.asap[des]))
                if len(common_des) > 0:
                    common_des.sort(key=lambda x: x[1])
                    # print(common_des)
                    start_node_dist[(a_node, b_node)] = (
                        common_des[0][1] * 2 - self.asap[a_node] - self.asap[b_node]
                    )
        # print(start_node_dist)

        final_str = ""
        for nodes, dist in start_node_dist.items():
            final_str += str(nodes[0]) + " " + str(nodes[1]) + " " + str(dist) + "\n"

        return final_str

    def set_node_feature(self):
        self.set_ASAP()
        self.set_in_degree()
        self.set_out_degree()
        self.set_no_grandchild()
        self.set_no_grandparent()
        self.set_no_descendant()
        self.set_no_ancestor()

    # below is for cgra-me
    def _get_descendants(self, node):
        descendants = set()
        stack = list(self.succ[node])
        while stack:
            n = stack.pop()
            if n not in descendants:
                descendants.add(n)
                stack.extend(self.succ[n])
        return descendants

    def _get_ancestors(self, node):
        ancestors = set()
        stack = list(self.pred[node])
        while stack:
            n = stack.pop()
            if n not in ancestors:
                ancestors.add(n)
                stack.extend(self.pred[n])
        return ancestors

    def _classify_nodes(self):
        loads = []
        stores = []
        outputs = []
        for nid in self.vertices:
            n_pred = len(self.pred[nid])
            has_succ = len(self.succ[nid]) > 0
            if n_pred == 1 and has_succ:
                loads.append(nid)
            elif n_pred == 2 and not has_succ:
                stores.append(nid)
            elif n_pred == 1 and not has_succ:
                outputs.append(nid)
        return loads, stores, outputs

    def _repair_excess_loads(self, loads, max_count):
        if len(loads) <= max_count:
            return
        excess = list(loads[max_count:])
        random.shuffle(excess)
        for nid in excess:
            descendants = self._get_descendants(nid)
            candidates = [
                n for n in self.vertices
                if n != nid
                and n not in descendants
                and n not in self.pred[nid]
                and len(self.succ[n]) > 0
            ]
            if not candidates:
                candidates = [
                    n for n in self.vertices
                    if n != nid
                    and n not in descendants
                    and n not in self.pred[nid]
                ]
            if candidates:
                new_pred = random.choice(candidates)
                self.edges.add((new_pred, nid))
                self.pred[nid].add(new_pred)
                self.succ[new_pred].add(nid)

    def _repair_excess_stores(self, stores, max_count):
        if len(stores) <= max_count:
            return
        excess = list(stores[max_count:])
        random.shuffle(excess)
        for nid in excess:
            ancestors = self._get_ancestors(nid)
            candidates = [
                n for n in self.vertices
                if n != nid
                and n not in ancestors
                and n not in self.succ[nid]
                and len(self.pred[n]) < 2
                and len(self.succ[n]) > 0
            ]
            if candidates:
                target = random.choice(candidates)
                self.edges.add((nid, target))
                self.succ[nid].add(target)
                self.pred[target].add(nid)

    def _repair_excess_outputs(self, outputs, max_count):
        if len(outputs) <= max_count:
            return
        node_num = len(self.vertices)
        max_store_count = int(0.1 * node_num)
        _, current_stores, _ = self._classify_nodes()
        store_budget = max_store_count - len(current_stores)

        excess = list(outputs[max_count:])
        random.shuffle(excess)
        for nid in excess:
            if nid not in self.vertices:
                continue
            if store_budget > 0:
                candidates = [
                    n for n in self.vertices
                    if n != nid and n not in self.pred[nid]
                ]
                if candidates:
                    new_pred = random.choice(candidates)
                    self.edges.add((new_pred, nid))
                    self.pred[nid].add(new_pred)
                    self.succ[new_pred].add(nid)
                    store_budget -= 1
                    continue
            for p in list(self.pred[nid]):
                self.succ[p].discard(nid)
                self.edges.discard((p, nid))
            del self.vertices[nid]
            del self.pred[nid]
            del self.succ[nid]

        for k in list(self.vertices.keys()):
            if len(self.succ[k]) == 0 and len(self.pred[k]) == 0:
                del self.vertices[k]
                del self.pred[k]
                del self.succ[k]

    def satisfy_cgra_me_constraint(self):
        # the number of input  node must be less or equal to 2
        # print("satisfy_cgra_me_constraint")
        # print("vertex:", self.vertices)
        # print("edges:",self.edges)
        # print("backtrack_edges:",self.backtrack_edges)
        # print("pred", self.pred)
        # print("succ", self.succ)

        # check input node limiation
        temp_pred = {}
        for node_id in self.vertices.keys():
            pred_limit = 2
            if len(self.succ[node_id]) == 0:
                pred_limit = randint(1, 2)
            if len(self.pred[node_id]) <= pred_limit:
                curr_pred = self.pred[node_id]
                curr_pred = list(curr_pred)
                temp_pred[node_id] = curr_pred
            else:
                diff = len(self.pred[node_id]) - pred_limit
                curr_pred = self.pred[node_id]
                curr_pred = list(curr_pred)
                sorted(curr_pred, key=lambda x: len(self.succ[x]))
                # print("befor", curr_pred)
                curr_pred = curr_pred[0:pred_limit]
                # print("after", curr_pred)
                temp_pred[node_id] = curr_pred
        # print("cgrame: temp_pred", temp_pred)
        self.edges.clear()
        self.pred.clear()
        self.succ.clear()

        for node in self.vertices.keys():
            self.succ[node] = set()
            self.pred[node] = set()

        for node_id, preds in temp_pred.items():
            for pred in preds:
                self.edges.add((pred, node_id))
                self.pred[node_id].add(pred)
                self.succ[pred].add(node_id)
        # print("cgrame: pred", self.pred)
        # print("cgrame: succ", self.succ)

        # remove isolated nodes (no predecessors and no successors after trimming)
        for k in list(self.vertices.keys()):
            if len(self.succ[k]) == 0 and len(self.pred[k]) == 0:
                del self.vertices[k]
                del self.pred[k]
                del self.succ[k]

        node_num = len(self.vertices.keys())
        if node_num == 0:
            return False

        max_loads = int(0.1 * node_num)
        max_stores = int(0.1 * node_num)
        max_outputs = int(0.2 * node_num)

        loads, stores, outputs = self._classify_nodes()

        self._repair_excess_loads(loads, max_loads)
        self._repair_excess_stores(stores, max_stores)

        loads, stores, outputs = self._classify_nodes()
        self._repair_excess_outputs(outputs, max_outputs)

        node_num = len(self.vertices.keys())
        if node_num == 0:
            return False
        loads, stores, outputs = self._classify_nodes()
        if float(len(loads)) / node_num > 0.1:
            return False
        if float(len(stores)) / node_num > 0.1:
            return False
        if float(len(outputs)) / node_num > 0.2:
            return False

        return True

    def assign_cgra_me_op_code(self):
        start_node = "const"
        two_op_general_op = ["add", "sub", "mul"]
        two_op_op_num = len(two_op_general_op) - 1
        for node_id in self.vertices.keys():
            succ = self.succ[node_id]
            pred = self.pred[node_id]
            if len(succ) == 0 and len(pred) == 2:
                self.vertices[node_id].opcode = "store"
            elif len(succ) == 0 and len(pred) == 1:
                self.vertices[node_id].opcode = "output"
                self.vertices[node_id].is_mem = 1
            elif len(pred) == 0:
                self.vertices[node_id].opcode = start_node
            elif len(pred) == 1:
                self.vertices[node_id].opcode = "load"
            elif len(pred) == 2:
                self.vertices[node_id].opcode = two_op_general_op[
                    random.randint(0, two_op_op_num)
                ]
            self.vertices[node_id].operand = len(succ)

    def cgrame_toStr(self):
        self.assign_cgra_me_op_code()
        final_str = ""
        for node_id in self.vertices.keys():
            node = self.vertices[node_id]
            final_str += node.opcode + str(node_id) + "[opcode=" + node.opcode + "]; \n"

        max_asap = 0
        for id, asap_ in self.asap.items():
            max_asap = max(max_asap, asap_)
        for target_asap in range(0, max_asap + 1):
            for nodeid, node in self.vertices.items():
                if self.asap[nodeid] is target_asap:
                    operand = 0
                    temp_name = node.opcode + str(int(node.id))
                    for pred in self.pred[nodeid]:
                        final_str += (
                            self.vertices[pred].opcode
                            + str(int(self.vertices[pred].id))
                            + "->"
                            + temp_name
                            + "[operand="
                            + str(operand)
                            + "];\n"
                        )
                        operand += 1

        return final_str

    # below is for morpher
    def check_morpher_edge_limit(self):
        # the number of input  node must be less or equal to 2
        # print("satisfy_cgra_me_constraint")
        # print("vertex:", self.vertices)
        # print("edges:",self.edges)
        # print("backtrack_edges:",self.backtrack_edges)
        # print("pred", self.pred)
        # print("succ", self.succ)

        # check input node limiation
        temp_pred = {}
        for node_id in self.vertices.keys():
            pred_limit = 3
            if len(self.succ[node_id]) == 0:
                pred_limit = randint(1, 3)
            if len(self.pred[node_id]) <= pred_limit:
                curr_pred = self.pred[node_id]
                curr_pred = list(curr_pred)
                temp_pred[node_id] = curr_pred
            else:
                diff = len(self.pred[node_id]) - pred_limit
                curr_pred = self.pred[node_id]
                curr_pred = list(curr_pred)
                sorted(curr_pred, key=lambda x: len(self.succ[x]))
                # print("befor", curr_pred)
                curr_pred = curr_pred[0:pred_limit]
                # print("after", curr_pred)
                temp_pred[node_id] = curr_pred
        # print("cgrame: temp_pred", temp_pred)
        self.edges.clear()
        self.pred.clear()
        self.succ.clear()

        for node in self.vertices.keys():
            self.succ[node] = set()
            self.pred[node] = set()

        for node_id, preds in temp_pred.items():
            for pred in preds:
                self.edges.add((pred, node_id))
                self.pred[node_id].add(pred)
                self.succ[pred].add(node_id)
        # print("cgrame: pred", self.pred)
        # print("cgrame: succ", self.succ)

        for k in list(self.vertices.keys()):
            if len(self.succ[k]) == 0 and len(self.pred[k]) == 0:
                del self.vertices[k]
                del self.pred[k]
                del self.succ[k]

        return True

    def assign_morpher_op_code(self):

        port_dest = ["I1", "I2", "P"]
        for node, vert in self.vertices.items():
            opcode_candidates = []
            if len(self.succ[node]) == 0:
                # store op
                # print("succ", len(self.succ[node]))
                # print("pred",len(self.pred[node]))
                opcode_candidates = morpher_store_opcode[len(self.pred[node])]
            else:
                opcode_candidates = morpher_nonstore_opcode[len(self.pred[node])]

            assert len(opcode_candidates) > 0
            vert.opcode = random.choice(opcode_candidates)
            # print("assign ", node, "to ", vert.opcode )

            index = 0
            for parent in self.pred[node]:
                self.vertices[parent].output_dest_port[node] = port_dest[index]
                index += 1

    def morpher_toStr(self):
        result = ""
        MutexBB = ET.Element("MutexBB")
        BB1 = ET.SubElement(MutexBB, "BB1", name="random")
        ET.SubElement(BB1, "BB2", name="random2")
        indent(MutexBB)
        result += ET.tostring(MutexBB).decode("utf-8")

        dfg = ET.Element("DFG", count=str(len(self.vertices)))
        dfg.tail = "\n"
        dfg.text = "\n"
        for index, vert in self.vertices.items():
            # print(index)
            node = ET.SubElement(
                dfg,
                "Node",
                idx=str(index),
                ASAP=str(self.asap[index]),
                ALAP="0",
                CONST="0",
                BB="random",
            )
            # node = ET.Element( "Node", idx=str(index), ASAP = str(self.asap[index]), ALAP = "0", CONST="0", BB="random")
            node.tail = "\n\n"
            node.text = "\n"

            op = ET.SubElement(node, "OP")
            op.text = vert.opcode
            op.tail = "\n"

            BasePointerName = ET.SubElement(node, "BasePointerName", size="1")
            BasePointerName.text = "random"
            BasePointerName.tail = "\n"

            inputs = ET.SubElement(node, "Inputs")
            inputs.tail = "\n"
            inputs.text = "\n"
            for parent in self.pred[index]:
                ET.SubElement(inputs, "MyInput", idx=str(parent)).tail = "\n"
            outpus = ET.SubElement(node, "Outputs")
            outpus.tail = "\n"
            outpus.text = "\n"
            for child, port in vert.output_dest_port.items():
                next_iter_value = "0"
                if self.asap[index] > self.asap[int(child)]:
                    next_iter_value = "1"
                if port == "P":
                    ET.SubElement(
                        outpus,
                        "MyOutput",
                        idx=str(child),
                        nextiter=next_iter_value,
                        NPB="0",
                        type=port,
                    ).tail = "\n"

                else:
                    ET.SubElement(
                        outpus,
                        "MyOutput",
                        idx=str(child),
                        nextiter=next_iter_value,
                        type=port,
                    ).tail = "\n"

            RecParents = ET.SubElement(node, "RecParents")
            RecParents.text = "\n"
            RecParents.tail = "\n"

        # indent(dfg)
        tree = ET.ElementTree(dfg)
        # ET.indent(tree, '\n')
        result += (ET.tostring(dfg, method="html")).decode("utf-8")

        return result

    def add_backedges(self):
        # actually, before this function, this one has a few backedges see in handle_cycle function.
        total_edge_num = len(self.edges)
        back_edge_candidates = []
        vertice_num = len(self.vertices)
        added_child_nodes = set()
        for i in range(vertice_num):
            for j in range(i + 1, vertice_num):
                if (
                    self.asap[i] > self.asap[j]
                    and len(self.pred[j]) < 3
                    and j not in added_child_nodes
                ):
                    added_child_nodes.add(j)
                    back_edge_candidates.append((i, j))
                    break
        be_candidate_num = len(back_edge_candidates)
        be_edge_num = ceil(total_edge_num / 10)
        # print("candidates num:", be_candidate_num)
        # print("edge num:", total_edge_num)
        # print("be edge num:", be_edge_num)
        # print("size", randomlist)
        # print("known_backedges, ", len(self.backedges))
        if len(self.backedges) > 0:
            return
        randomlist = random.sample(range(0, be_candidate_num - 1), be_edge_num)

        for i in range(be_edge_num):
            src, des = back_edge_candidates[randomlist[i]]
            self.backedges.add((src, des))
            self.pred[des].add(src)
            assert len(self.pred[des]) <= 3
            self.succ[src].add(des)

        assert be_edge_num == len(self.backedges)


if __name__ == "__main__":
    MIN_NODE = 20
    MAX_NODE = 30

    min_asap = 0

    number_node = random.choice(range(MIN_NODE, MAX_NODE))
    min_edge = 2  # for each node
    max_edge = random.choice(range(3, 5))  # for each node

    while True:
        edge_dic = dfg_json_maker(
            str(1), 0, 0, number_node, min_edge, max_edge, 0, 1, 2, 1
        )
        graph = DFGGraph("test")
        for num in range(number_node):
            graph.add_vertex(Vertex(id=str(num + 1)))

        for key, values in edge_dic.items():
            for value in values:
                graph.add_edge(key, value)

        node_number = len(graph.vertices.keys())
        graph.check_connectivity()
        graph.handle_cycle()
        if len(graph.vertices) == 0:
            continue

        if not graph.check_morpher_edge_limit():
            # print("did not generate", i)
            continue
        if not graph.check_connectivity():
            continue

        new_node_number = len(graph.vertices.keys())
        # if node_number != new_node_number:
        # add somework to handle it
        graph.make_node_index_continous(node_number)
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

        graph.add_backedges()
        graph.assign_morpher_op_code()

        if len(graph.vertices) == 0:
            continue

        # in_degree = graph.get
        # out_degree = graph.get_out_degree()

        result = graph.morpher_toStr()

        f = open("filename.xml", "w")
        f.write(result)
        f.close()

        print("asap", graph.asap)
        print("edge", graph.edges)
        print("backedges", graph.backedges)
        # print("asap_value", asap_value)

        sys.exit()
    # print("labels", labels)
    # print("in_degree", in_degree)
    # print("out_degree", out_degree)
