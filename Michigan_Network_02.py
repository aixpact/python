# coding: utf-8

# # Assignment 2 - Network Connectivity
# In this assignment you will go through the process of importing and analyzing an internal email communication network between employees of a mid-sized manufacturing company.
# Each node represents an employee and each directed edge between two nodes represents an individual email. The left node represents the sender and the right node represents the recipient.

import networkx as nx

# This line must be commented out when submitting to the autograder
# !head email_network.txt


# ### Question 1
# Using networkx, load up the directed multigraph from `email_network.txt`. Make sure the node names are strings.
# *This function should return a directed multigraph networkx graph.*
def answer_one():
    # Your Code Here
    G = nx.read_edgelist('email_network.txt', data=[('time', int)], create_using=nx.MultiDiGraph())
    assert G.is_directed()
    assert G.is_multigraph()

    return G

answer_one()


# ### Question 2
# How many employees and emails are represented in the graph from Question 1?
# *This function should return a tuple (#employees, #emails).*
def answer_two():
    G = answer_one()

    # Frequency
    frequency = G.degree()

    employees = len(G.nodes())
    emails = len(G.edges())

    return employees, emails

answer_two()


# ### Question 3
# * Part 1. Assume that information in this company can only be exchanged through email.
#     When an employee sends an email to another employee, a communication channel has been created,
#     allowing the sender to provide information to the receiver, but not vice versa.
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# * Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways.
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# *This function should return a tuple of bools (part1, part2).*
def answer_three():
    G = answer_one()

    # 1. Strong connected?
    strong = nx.is_strongly_connected(G)

    # 2. Weak connected?
    weak = nx.is_weakly_connected(G)

    return strong, weak

answer_three()


# ### Question 4
# How many nodes are in the largest (in terms of nodes) weakly connected component?
# *This function should return an int.*
def answer_four():
    # Your Code Here
    G = answer_one()
    no_weak_components = nx.number_weakly_connected_components(G)
    no_weak_nodes = len(list(nx.weakly_connected_components(G))[0])

    return no_weak_nodes

answer_four()


# ### Question 5
# How many nodes are in the largest (in terms of nodes) strongly connected component?
# *This function should return an int*
def answer_five():
    # Your Code Here
    G = answer_one()
    #     no_strong_components = nx.number_strongly_connected_components(G)

    components = sorted(nx.strongly_connected_components(G))
    max_nodes_per_component = max([len(c) for c in components])

    # or
    #     components = nx.strongly_connected_component_subgraphs(G)
    #     max_nodes_per_component = max([len(c) for c in components])

    return max_nodes_per_component

answer_five()


# ### Question 6
# Using the NetworkX function strongly_connected_component_subgraphs, find the subgraph of nodes in a largest strongly connected component.
# Call this graph G_sc.
# *This function should return a networkx MultiDiGraph named G_sc.*
def answer_six():
    G = answer_one()
    G_sc = [Gc for Gc in sorted(nx.strongly_connected_component_subgraphs(G), key=len, reverse=True)][0]

    return G_sc

answer_six()


# ### Question 7
# What is the average distance between nodes in G_sc?
# *This function should return a float.*
def answer_seven():
    G_sc = answer_six()

    return nx.average_shortest_path_length(G_sc)

answer_seven()


# ### Question 8
# What is the largest possible distance between two employees in G_sc?
# *This function should return an int.*
def answer_eight():
    G_sc = answer_six()

    return nx.diameter(G_sc)

answer_eight()


# ### Question 9
# What is the set of nodes in G_sc with eccentricity equal to the diameter?
# *This function should return a set of the node(s).*
def answer_nine():
    # Your Code Here
    G_sc = answer_six()

    return set(nx.periphery(G_sc))

answer_nine()


# ### Question 10
# What is the set of node(s) in G_sc with eccentricity equal to the radius?
# *This function should return a set of the node(s).*
def answer_ten():
    G_sc = answer_six()

    return set(nx.center(G_sc))

answer_ten()


# ### Question 11
# Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?
# How many nodes are connected to this node?
# *This function should return a tuple (name of node, number of satisfied connected nodes).*
def answer_eleven():
    # all nodes have max distance of diameter!
    # a. #connections with diameter-length paths
    # b. node[a] with max connections
    from collections import Counter

    G_sc = answer_six()
    peri_nodes = nx.periphery(G_sc)
    diameter = nx.diameter(G_sc)

    # max(frequency of values == diameter per node)
    return max([
        (node, Counter(nx.shortest_path_length(G_sc, node).values())[diameter])
        for node in peri_nodes])

    # same as:
    # max_count = -1
    # result_node = None
    # for node in peri_nodes:
    #     sp = nx.shortest_path_length(G, node)
    #     count = list(sp.values()).count(diameter)
    #     if count > max_count:
    #         result_node = node
    #         max_count = count
    # return result_node, max_count

answer_eleven()




# ### Question 12
# Suppose you want to prevent communication from flowing to the node that you found in the previous question from any node in the center of G_sc, what is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous question or the center nodes)?
# *This function should return an integer.*
def answer_twelve():
    # Your Code Here
    G_sc = answer_six()
    node_11 = answer_eleven()[0]
    center_nodes = nx.center(G_sc)

    no_cut_nodes = len([nx.minimum_node_cut(G_sc, cn, node_11) for cn in center_nodes][0])

    return no_cut_nodes

answer_twelve()


# ### Question 13
# Construct an undirected graph G_un using G_sc (you can ignore the attributes).
# *This function should return a networkx Graph.*
def answer_thirteen():
    # Your Code Here
    G_sc = answer_six()
    assert G_sc.is_multigraph() | G_sc.is_directed()

    G_un = nx.Graph(G_sc).to_undirected()
    #     assert G_un.is_multigraph() & G_un.is_directed()

    return G_un

answer_thirteen()


# ### Question 14
# What is the transitivity and average clustering coefficient of graph G_un?
# *This function should return a tuple (transitivity, avg clustering).*
def answer_fourteen():
    # Your Code Here
    G_un = answer_thirteen()

    transitivity = nx.transitivity(G_un)
    avg_clustering = nx.average_clustering(G_un)

    return transitivity, avg_clustering

answer_fourteen()
