# -*- coding: utf-8 -*-
#
#    Copyright (C) 2018 by
#    Thomas Bonald <thomas.bonald@telecom-paristech.fr>
#    Bertrand Charpentier <bertrand.charpentier@live.fr>
#    All rights reserved
#    BSD license

import networkx as nx
import numpy as np

## Aggregation functions

def init_aggregate_graph(graph):
    """Init the graph to be aggregated

     Parameters
     ----------
     graph : networkx.graph
         An undirected graph with weighted edges (default weight = 1)
         Nodes must be numbered from 0 to n - 1 (n nodes)

     Returns
     -------
     aggregate_graph : networkx.graph
         Same graph (copied) without self-loops and with node attributes (weight, size = 1)
     total_weight : int
         Total weight of nodes (twice the total weight of edges)
    """

    aggregate_graph = graph.copy()
    
    # remove self-loops, add node weights
    node_weights = {u: 0. for u in aggregate_graph.nodes()}
    total_weight = 0. 
    edges = list(aggregate_graph.edges())
    for (u,v) in edges:
        if u == v:
            aggregate_graph.remove_edge(u,u)
        else:
            if 'weight' not in aggregate_graph[u][v]:
                aggregate_graph[u][v]['weight'] = 1.
            weight = aggregate_graph[u][v]['weight']
            node_weights[u] += weight
            node_weights[v] += weight
            total_weight += 2 * weight
    nx.set_node_attributes(aggregate_graph, node_weights, 'weight')
    
    # add node sizes   
    nx.set_node_attributes(aggregate_graph, 1, 'size')
    return aggregate_graph, total_weight

def merge_nodes(graph, u, v, new_node):
    """Merge two nodes of a graph and update the graph

     Parameters
     ----------
     graph : networkx.graph
         An undirected graph with node attributes (weight, size)
         Nodes must be numbered from 0 to n - 1 (n nodes)

     Returns
     -------
     grap h: networkx.graph
         Same graph with nodes u and v replaced by node new_node
     u,v : int
         Nodes to be merged
     new_node : int
         Node replacing u,v (with total weight and total size of u and v)
    """
    
    neighbors_u = list(graph.neighbors(u))
    neighbors_v = list(graph.neighbors(v))
    graph.add_node(new_node)

    # update edges
    for node in neighbors_u:
        graph.add_edge(new_node,node,weight = graph[u][node]['weight'])
    for node in neighbors_v:
        if graph.has_edge(new_node,node):
            graph[new_node][node]['weight'] += graph[v][node]['weight']
        else:
            graph.add_edge(new_node,node,weight = graph[v][node]['weight'])

    # updage node attributes
    graph.node[new_node]['weight'] = graph.node[u]['weight'] + graph.node[v]['weight']
    graph.node[new_node]['size'] = graph.node[u]['size'] + graph.node[v]['size']
    graph.remove_node(u)
    graph.remove_node(v)
    return graph

## Clustering algorithms

def paris_hierarchy(graph):
    """Paris agglomerative algorithm

     Parameters
     ----------
     graph : networkx.graph
         An undirected graph with weighted edges (default weight = 1)
         Nodes must be numbered from 0 to n - 1 (n nodes)

     Returns
     -------
     dendrogram : numpy.ndarray
         2D array
         Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster

     Reference
     ---------
     T. Bonald, B. Charpentier, A. Galland, A. Hollocou, Hierarchical graph clustering using node pair sampling, KDD Workshop, 2018     
     """
    
    # dendrogram as list of merges
    dendrogram = []
    
    aggregate_graph, total_weight = init_aggregate_graph(graph)
    number_nodes = aggregate_graph.number_of_nodes()
    new_node = number_nodes
    while number_nodes > 0:
        # nearest-neighbor chain
        chain = [list(aggregate_graph.nodes())[0]]
        while chain != []:
            current_node = chain.pop()
            # find nearest neighbor 
            distance_min = float("inf")
            nearest_neighbor = -1
            for node in aggregate_graph.neighbors(current_node):
                if node != current_node:
                    distance = (aggregate_graph.node[current_node]['weight'] * aggregate_graph.node[node]['weight'] 
                        / aggregate_graph[current_node][node]['weight'] / total_weight)
                    if distance < distance_min:
                        nearest_neighbor = node
                        distance_min = distance
                    elif distance == distance_min:
                        nearest_neighbor = min(nearest_neighbor,node)
            distance = distance_min
            if chain != []:
                next_node = chain.pop()
                if next_node == nearest_neighbor:
                     # merge nodes
                    size = aggregate_graph.node[current_node]['size'] + aggregate_graph.node[next_node]['size']
                    dendrogram.append([current_node,next_node,distance,size])
                    aggregate_graph = merge_nodes(aggregate_graph,current_node,next_node,new_node)
                    number_nodes -= 1
                    new_node += 1
                else:
                    chain.append(next_node)
                    chain.append(current_node)
                    chain.append(nearest_neighbor)
            elif nearest_neighbor >= 0:
                chain.append(current_node)
                chain.append(nearest_neighbor)
            else:
                number_nodes -= 1
    return np.array(dendrogram, float)


def newman_hierarchy(graph):
    """Newman's agglomerative algorithm

     Parameters
     ----------
     graph : networkx.graph
         An undirected graph with weighted edges (default weight = 1)
         Nodes must be numbered from 0 to n - 1 (n nodes)

     Returns
     -------
     dendrogram : numpy.ndarray
         2D array
         Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster

     Reference
     ---------
     M. E. Newman (2004). Fast algorithm for detecting community structure in networks. Physical review E. 
     """
    
    # dendrogram as list of merges
    dendrogram = []
    
    aggregate_graph, total_weight = init_aggregate_graph(graph)
    # modularity increase
    for (u,v) in aggregate_graph.edges():
        aggregate_graph[u][v]['delta'] = 2 * (aggregate_graph[u][v]['weight']
                                            - aggregate_graph.node[u]['weight'] 
                                            * aggregate_graph.node[v]['weight'] 
                                            / total_weight) / total_weight
        
    number_nodes = aggregate_graph.number_of_nodes()
    new_node = number_nodes
    while number_nodes > 1:
        # find the best node pair for modularity increase
        delta = nx.get_edge_attributes(aggregate_graph,'delta')
        u,v = max(delta, key = delta.get)
        # merge nodes
        size = aggregate_graph.node[u]['size'] + aggregate_graph.node[v]['size']
        dendrogram.append([u,v,size,size])
        aggregate_graph = merge_nodes(aggregate_graph,u,v,new_node)
        for u in aggregate_graph.neighbors(new_node):
            aggregate_graph[u][new_node]['delta'] = 2 * (aggregate_graph[u][new_node]['weight']  
                                              - aggregate_graph.node[u]['weight'] 
                                                * aggregate_graph.node[new_node]['weight'] 
                                              / total_weight) / total_weight
        number_nodes -= 1
        new_node += 1
    return np.array(dendrogram, float)


def random_hierarchy(graph):
    """Random agglomerative algorithm

     Parameters
     ----------
     graph : networkx.graph
         An undirected graph with weighted edges (default weight = 1)
         Nodes must be numbered from 0 to n - 1 (n nodes)

     Returns
     -------
     dendrogram : numpy.ndarray
         2D array
         Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster
     """
    
    # dendrogram as list of merges
    dendrogram = []
    
    aggregate_graph, total_weight = init_aggregate_graph(graph)

    number_nodes = aggregate_graph.number_of_nodes()
    new_node = number_nodes
    while number_nodes > 1:
        # random edge 
        edges = list(aggregate_graph.edges())
        u,v = edges[np.random.randint(len(edges))]
        # merge nodes
        size = aggregate_graph.node[u]['size'] + aggregate_graph.node[v]['size']
        dendrogram.append([u,v,size,size])
        aggregate_graph = merge_nodes(aggregate_graph,u,v,new_node)
        number_nodes -= 1
        new_node += 1

    return np.array(dendrogram, float)

def hierarchical_clustering(graph, algorithm):
    """Hierarchical clustering

     Parameters
     ----------
     graph : networkx.graph
         An undirected graph with weighted edges (default weight = 1)
         Nodes must be numbered from 0 to n - 1 (n nodes)
         The graph must be connected
     algorithm : {"paris","newman","random"}
         Clustering algorithm

     Returns
     -------
     dendrogram : numpy.ndarray
         2D array
         Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster
    """
    
    number_nodes = graph.number_of_nodes()
    if set(graph.nodes()) != set(range(number_nodes)):
        print("Error: Nodes must be numbered from 0 to n - 1.\nYou may consider the networkx function 'convert_node_labels_to_integers'.")
    elif not nx.is_connected(graph):
        print("Error: The graph is not connected.\nYou may consider the networkx function 'connected_component_subgraphs'.")
    else:
        if algorithm == "paris":
            return paris_hierarchy(graph)
        elif algorithm == "newman":
            return newman_hierarchy(graph)
        elif algorithm == "random":
            return random_hierarchy(graph)
        else:
            print("Unknown algorithm")

## Metrics

def relative_entropy(graph, dendrogram, weighted = True):
    """Relative entropy of a hierarchy (quality metric)

     Parameters
     ----------
     graph : networkx.graph
         An undirected graph with weighted edges (default weight = 1)
         Nodes must be numbered from 0 to n - 1 (n nodes)
         The graph must be connected
     dendrogram : numpy.ndarray
         2D array
         Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster
     weighted : bool, optional
         The reference node sampling distribution is proportional to the weights if True and uniform if False

     Returns
     -------
     quality : float
         The relative entropy of the hierarchy (quality metric)

     Reference
     ---------
     T. Bonald, B. Charpentier (2018), Learning Graph Representations by Dendrograms, https://arxiv.org/abs/1807.05087
    """
    
    aggregate_graph, total_weight = init_aggregate_graph(graph) 
    number_nodes = aggregate_graph.number_of_nodes()
    if weighted:
        pi = {u: aggregate_graph.node[u]['weight'] / total_weight for u in aggregate_graph.nodes()}
    else:
        pi = {u: 1. / number_nodes for u in aggregate_graph.nodes()}
    quality = 0.
    for t in range(number_nodes - 1):
        u = int(dendrogram[t][0])
        v = int(dendrogram[t][1])
        if aggregate_graph.has_edge(u,v):
            p = 2 * aggregate_graph[u][v]['weight'] / total_weight 
            quality += p * np.log(p / pi[u] / pi[v])
        aggregate_graph = merge_nodes(aggregate_graph, u, v, number_nodes + t)
        pi[number_nodes + t] = pi.pop(u) + pi.pop(v)
    return quality

def dasgupta_cost(graph, dendrogram, weighted = True):    
    """Dasgupa's cost of a hierarchy (cost function)

     Parameters
     ----------
     graph : networkx.graph
         An undirected graph with weighted edges (default weight = 1)
         Nodes must be numbered from 0 to n - 1 (n nodes)
         The graph must be connected
     dendrogram : numpy.ndarray
         2D array
         Each row contains the two merged nodes, the height in the dendrogram, and the size of the corresponding cluster
     weighted : bool, optional
         The reference node sampling distribution is proportional to the weights if True and uniform if False

     Returns
     -------
     cost : float
         Dasgupta's cost function (cost)

     Reference
     ---------
     S. Dasgupta (2016). A cost function for similarity-based hierarchical clustering. In Proceedings of ACM symposium on Theory of Computing.
    """
    
    aggregate_graph, total_weight = init_aggregate_graph(graph) 
    number_nodes = aggregate_graph.number_of_nodes()
    if weighted:
        pi = {u: aggregate_graph.node[u]['weight'] / total_weight for u in aggregate_graph.nodes()}
    else:
        pi = {u: 1. / number_nodes for u in aggregate_graph.nodes()}
    cost = 0.
    for t in range(number_nodes - 1):
        u = int(dendrogram[t][0])
        v = int(dendrogram[t][1])
        if aggregate_graph.has_edge(u,v):
            p = 2 * aggregate_graph[u][v]['weight'] / total_weight 
            cost += p * (pi[u] + pi[v])
        aggregate_graph = merge_nodes(aggregate_graph, u, v, number_nodes + t)
        pi[number_nodes + t] = pi.pop(u) + pi.pop(v)
    return cost    
