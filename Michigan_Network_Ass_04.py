import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import operator

# ---------------------------------------------> 4.1 - preferential wiring <-------------------------------------------- #
# Many real (world) networks have degree distributions that look like Power Law - (𝑃𝑘 = C𝑘^α) (typical α ± 2-4)
# Degree distribution of a graph/network == probability distribution of the degrees over the entire network
# Preferential Attachment Model produces networks with a power law degree distribution


# Probability of connecting to a node μ of degree k_μ is (k_μ / sum(k_γ))
# Probability of node is relative degree: degrees_node/total_degree_graph)
# Attachment new node according to this probability distribution

# Degree distribution = relative frequency of (in-)degrees in graph

# barabasi_albert_graph(n, m):
#  - n-node preferential attachment network, where:
#  - each new node attaches to m existing nodes
G = nx.barabasi_albert_graph(100000, 1)
degrees = G.degree()
dict_degrees = {k: v for k, v in sorted(degrees)}
degree_set = sorted(set(dict_degrees.values()))
degree_list = list(dict_degrees.values())

histogram = [degree_list.count(i)/float(nx.number_of_nodes(G)) for i in degree_set]
plt.plot(degree_set, histogram, 'o')
plt.xlabel('Degree')
plt.ylabel('Fraction of Nodes')
plt.xscale('log')
plt.yscale('log')
plt.show()

# ---------------------------------------------> 4.2 - rewiring <-------------------------------------------- #

# Social networks tend to have high clustering coefficient and small average path length
G = nx.barabasi_albert_graph(1000, 6)

# Average Clustering Coefficient(Average CC)
# Local Clustering Coefficient(Local CC): Fraction of pairs of the node’s friends that are friends with each other.
print(nx.average_clustering(G))

# Average path length - n-degrees of separation
# median path length: typically between 4-7
print(nx.average_shortest_path_length(G))

# Path length distribution

# Motivation: Real networks exhibit high clustering coefficient and small average shortest paths.
# Can we think of a model that achieves both of these properties?

# a. Regular Lattice (𝑝 = 0): no edge is rewired.
# b. Random Network (𝑝 = 1): all edges are rewired
# c. Small World Network (0 < 𝑝 < 1): Some edges are rewired. Network conserves some local structure but has some randomness

# watts_strogatz_graph(n, k, p) returns a small world network with
#   n nodes
#   starting with a ring lattice with each node connected to its k nearest neighbors,
#   rewiring probability p
G = nx.watts_strogatz_graph(1000, 6, 0.04)
degrees = G.degree()

dict_degrees = {k: v for k, v in sorted(degrees)}
degree_set = sorted(set(dict_degrees.values()))
degree_list = list(dict_degrees.values())

histogram = [degree_list.count(i)/float(nx.number_of_nodes(G)) for i in degree_set]
plt.bar(degree_set, histogram)
plt.xlabel('Degree')
plt.ylabel('Fraction of Nodes')
plt.show()

# No power law degree distribution:
#   Since most edges are not rewired, most nodes have degree of 6.
#   Since edges are rewired uniformly at random, no node accumulated very high degree, like in the preferential attachment model
plt.plot(degree_set, histogram, 'o')
plt.xlabel('Degree')
plt.ylabel('Fraction of Nodes')
plt.xscale('log')
plt.yscale('log')
plt.show()

G = nx.connected_watts_strogatz_graph(n, k, p, t)
#  runs watts_strogatz_graph(n, k, p) up to t times, until it returns a connected small world network

G = nx.newman_watts_strogatz_graph(n, k, p)
# runs a model similar to the small world model, but rather than rewiring edges, new edges are added with probability 𝑝

#  Summary:
# • Real social networks:
#  - small shortest paths  and
#  - high clustering coefficient.
# • The preferential attachment model:
#  - small shortest paths
#  - very small clustering
# • The small world model(p = small):
#  - small average shortest path
#  - high clustering coefficient, => matching real networks. However, the degree distribution is not a power law.
# • watts_strogatz_graph(n, k, p) (and other variants) to produce small world networks

# ---------------------------------------------> 4.3 Link Prediction <-------------------------------------------- #

# Link prediction: Given a network, can we predict which edges will be formed in the future?
# - What new edges are likely to form in this network?
# - Given a pair of nodes, how to assess whether they are likely to connect?

# Triadic closure: the tendency for people who share connections in a social network to become connected.

# Measure 1: Common Neighbors (intercept)
# The number of common neighbors of nodes 𝑋 and 𝑌
G = nx.newman_watts_strogatz_graph(100, 5, 0.1)
common_neigh = [(e[0], e[1], len(list(nx.common_neighbors(G, e[0], e[1])))) for e in nx.non_edges(G)]
sorted(common_neigh, key=operator.itemgetter(2), reverse=True); common_neigh

# Measure 2: Jaccard Coefficient (intercept over union)
# Number of common neighbors normalized by the total number of neighbors
# common_neighbors/total_neighbors
L = list(nx.jaccard_coefficient(G))
L.sort(key=operator.itemgetter(2), reverse=True); L

# Measure 3: Resource
# Fraction of a ”resource” that a node can send to another through their common neighbors
# sum(1/degree_common_neighbor)
L = list(nx.resource_allocation_index(G))
L.sort(key=operator.itemgetter(2), reverse=True); L

# Measure 4:
# Adamic Adar Index
# Similar to resource allocation index, but with log in the denominator
# sum(1/log(degree_common_neighbor))
L = list(nx.adamic_adar_index(G))
L.sort(key=operator.itemgetter(2), reverse=True); L

# Method 5:
# Preferential Attachment
# In the preferential attachment model, nodes with high degree get more neighbors
# degree_source * degree_target
L = list(nx.preferential_attachment(G))
L.sort(key=operator.itemgetter(2), reverse=True); L

# Measure 6:
# Community Common Neighbors
# Number of common neighbors with bonus of 1 for each neighbor in same community
# f(u) = 1 if same community else 0
# sum(f(u) * degree)
G = nx.newman_watts_strogatz_graph(9, 5, 0.1)
G.nodes()
G.node[0]['community'] = 0
G.node[1]['community'] = 0
G.node[2]['community'] = 0
G.node[3]['community'] = 0
G.node[4]['community'] = 1
G.node[5]['community'] = 1
G.node[6]['community'] = 1
G.node[7]['community'] = 1
G.node[8]['community'] = 1
L = list(nx.cn_soundarajan_hopcroft(G))
L.sort(key=operator.itemgetter(2), reverse=True); L

# Measure 7:
# Community Resource Allocation
# Similar to resource allocation index, but only considering nodes in the same community
# f(u) = 1 if same community else 0
# sum(f(u)/degree)
L = list(nx.ra_index_soundarajan_hopcroft(G))
L.sort(key=operator.itemgetter(2), reverse=True); L

# Summary
# • Link prediction problem: Given a network, predict which edges will be formed in the future.
# • 5 basic measures:
# – NumberofCommonNeighbors – JaccardCoefficient
# – ResourceAllocationIndex
# – Adamic-AdarIndex
# – PreferentialAttachmentScore
# • 2 measures that require community information:
# – CommonNeighborSoundarajan-HopcroftScore – ResourceAllocationSoundarajan-HopcroftScore


# ---------------------------------------------> Plot <-------------------------------------------- #

# draw the graph using the default spring layout
# See what layouts are available in networkX
[x for x in nx.__dir__() if x.endswith('_layout')]

plt.figure(figsize=(12, 8))
pos_a = nx.random_layout(G)
pos_b = nx.get_node_attributes(G, 'location') # for nodeswith geo location attribute
plt.axis('off')
nx.draw_networkx(G, pos_a, alpha=0.7, with_labels=False, edge_color='.4')

# Draw graph with varying node color, node size, and edge width
plt.figure(figsize=(12, 8))
pos_a = nx.random_layout(G)
node_color =  [1000*nx.degree_centrality(G)[v] for v in G] # [50*nx.degree(G)[v] for v in G]
node_size = [100*G.degree(v) for v in G]
edge_width = [10*nx.betweenness_centrality(G, normalized = True, endpoints=False)[v] for v in G]
# 0.5 or [0.0015*G[u][v]['weight'] for u, v in G.edges()]
nx.draw_networkx(G, pos_a, node_size=node_size,
                 node_color=node_color, alpha=0.7, with_labels=True,
                 width=edge_width, edge_color='.4', cmap=plt.cm.Blues)
# nx.draw_networkx_labels(G, pos_a, labels={'1': '1', '2': '2'}, font_size=18, font_color='w')
plt.axis('off')
plt.tight_layout()

# ---------------------------------------------> Assignment 4 <-------------------------------------------- #

import networkx as nx
import pandas as pd
import numpy as np
import pickle

P1_Graphs = pickle.load(open('A4_graphs.dms','rb'))
P1_Graphs

def graph_identification():
    #     import matplotlib as mpl
    #     import matplotlib.pyplot as plt

    # Your Code Here
    algs = ['PA', 'SW_L', 'SW_H']

    # Real World: degree dist is Power Law
    # PA: Power Law; log/log degree distribution is straight line
    # => max clustering = 0.1, max shortest path = 7,5

    # Small world: degree dist is not Power Law
    # More nodes: higher Avg shortest path, Avg lower clustering
    # Higher rewiring p: lower clustering and lower shortest path
    # - SW_L: Lattice - Small World; higher shortest paths > 7.5, higher clustering
    # - SW_H: Small world - Random;  lower shortest paths < 7.5, lower clustering
    # SW_L: => max clustering = 0.1, max shortest path = 7,5
    # SW_H: => max clustering = 0.02, max shortest path = 4,5

    def graph_algorithm_params(G):

        degrees = G.degree()
        #         print(len(degrees))
        degree_set = sorted(set(degrees.values()))
        degree_list = list(degrees.values())
        histogram = [degree_list.count(i) / float(nx.number_of_nodes(G)) for i in degree_set]
        print(nx.average_clustering(G), nx.average_shortest_path_length(G), len(histogram))
        if histogram[0] > histogram[1] > histogram[2] > histogram[3]:
            print('PA')
        elif (nx.average_shortest_path_length(G) < 7.5) | (nx.average_clustering(G) < 0.25):
            print('SW_H')
        else:
            print('SW_L')
        plt.bar(degree_set, histogram)
        plt.xlabel('Degree')
        plt.ylabel('Fraction of Nodes')
        plt.show()

    #     graph_algorithm_params(nx.barabasi_albert_graph(1000, 1))
    #     graph_algorithm_params(nx.barabasi_albert_graph(1000, 2))
    #     graph_algorithm_params(nx.barabasi_albert_graph(1000, 5))
    #     graph_algorithm_params(nx.barabasi_albert_graph(1000, 20))

    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 2, 0.1, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 5, 0.1, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 10, 0.1, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 2, 0.3, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 6, 0.25, 20))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 2, 0.7, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 5, 0.7, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 12, 1, 10))

    #     for graph in P1_Graphs:
    #         graph_algorithm_params(graph)

    algs = ['PA', 'SW_L', 'SW_L', 'PA', 'SW_H']

    return algs

graph_identification()



G = nx.read_gpickle('email_prediction.txt')

G.edges()
print(nx.info(G))
G.nodes(data=True)


def salary_predictions():
    # Import preprocessing, selection and metrics
    from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
    from sklearn.metrics import roc_auc_score
    from sklearn.dummy import DummyClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC

    # Your Code Here
    G.edges(data=True)

    df = pd.DataFrame(index=G.nodes())
    df['Department'] = pd.Series(nx.get_node_attributes(G, 'Department'))
    df['ManagementSalary'] = pd.Series(nx.get_node_attributes(G, 'ManagementSalary'))
    df['clustering'] = pd.Series(nx.clustering(G))
    df['degree'] = pd.Series(G.degree())
    df['closeness'] = pd.Series(nx.closeness_centrality(G, normalized=True))
    df['betweenness'] = pd.Series(nx.betweenness_centrality(G, normalized=True))
    df['pagerank'] = pd.Series(nx.pagerank(G, alpha=0.80))
    df['hub'] = pd.Series(nx.hits(G)[0])
    df['authority'] = pd.Series(nx.hits(G)[1])

    df_test_mask = pd.isnull(df.loc[:, 'ManagementSalary'])
    df_train = df.loc[~df_test_mask, :]
    df_test = df.loc[df_test_mask, :]

    #
    y_train = df_train.pop('ManagementSalary')
    X_train = df_train
    df_test.drop('ManagementSalary', axis=1, inplace=True)
    X_test = df_test

    def auc_scores(model, *args, k=5, threshold=0.50):
        """CV scores"""
        X, y = args
        predictions = cross_val_predict(model, X, y, cv=k, n_jobs=-1)
        print('AUC - Test predict  {:.2%}'.format(roc_auc_score(y, predictions)))
        print('AUC - Test probabil {:.2%}'.format(roc_auc_score(y, pred_probas)))

    classifiers = [
        #         GaussianNB(),
        #         DecisionTreeClassifier(random_state=0),
        #         DecisionTreeClassifier(max_depth=3, random_state=0),
        #         DecisionTreeClassifier(max_depth=4, random_state=0),
        #         DecisionTreeClassifier(max_depth=5, random_state=0),
        #         DecisionTreeClassifier(max_depth=6, random_state=0),
        GradientBoostingClassifier(random_state=0),
        #         GradientBoostingClassifier(learning_rate=0.08, random_state=0),
        #         GradientBoostingClassifier(learning_rate=0.12, random_state=0),
        #         GradientBoostingClassifier(learning_rate=0.1, max_depth=3, random_state=0),
        #         GradientBoostingClassifier(learning_rate=0.1, max_depth=4, random_state=0),
        #         RandomForestClassifier(n_estimators=100, random_state=0),
        #         AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=0),
        #         KNeighborsClassifier(),
        #         KNeighborsClassifier(n_neighbors=4),
        #         LinearSVC(random_state=0)
    ]

    for model in classifiers:
        #         print('-'*80)
        #         print(model)

        # Training scores
        #         clf_train = model.fit(X_train, y_train)
        #         pred_train = clf_train.predict(X_train)
        #         print('AUC - Train pred    {:.2%}'.format(roc_auc_score(y_train, pred_train)))

        # CV scores
        clf = model.fit(X_train, y_train)
    #         auc_scores(clf, X_train, y_train)

    # Predict
    #     predicted = [x for x, y in clf.predict_proba(X_test)]
    predicted = pd.DataFrame(clf.predict_proba(X_test), columns=clf.classes_)
    print(predicted)
    pred_series = pd.Series(predicted)
    assert type(pred_series) == pd.Series, 'wtf: ' + str(type(pred_series))

    return pred_series

salary_predictions()


def new_connections_predictions():
    import operator
    # Import preprocessing, selection and metrics
    from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
    from sklearn.metrics import roc_auc_score
    from sklearn.dummy import DummyClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC

    # Your Code Here
    df_fc_test_mask = pd.isnull(future_connections.loc[:, 'Future Connection'])
    df = pd.DataFrame()

    # Measure 1: Common Neighbors (intercept)
    # The number of common neighbors of nodes 𝑋 and 𝑌
    #     future_connections['common_neigh']
    L = [(e[0], e[1], len(list(nx.common_neighbors(G, e[0], e[1]))))
         for e in nx.non_edges(G)]
    df['pair'] = [(x, y) for x, y, z in L]
    df['common_nb'] = [z for x, y, z in L]
    #     L.sort(key=operator.itemgetter(2), reverse=True)
    #     print(L)

    # Measure 2: Jaccard Coefficient (intercept over union)
    # Number of common neighbors normalized by the total number of neighbors
    # common_neighbors/total_neighbors
    #     future_connections['jaccard']
    df['jaccard'] = pd.Series([z for x, y, z in nx.jaccard_coefficient(G)])
    #     L.sort(key=operator.itemgetter(2), reverse=True)
    #     print(L)

    # Measure 3: Resource
    # Fraction of a ”resource” that a node can send to another through their common neighbors
    # sum(1/degree_common_neighbor)
    df['resource'] = pd.Series([z for x, y, z in nx.resource_allocation_index(G)])
    #     L.sort(key=operator.itemgetter(2), reverse=True)
    #     print(L)

    # Measure 4:
    # Adamic Adar Index
    # Similar to resource allocation index, but with log in the denominator
    # sum(1/log(degree_common_neighbor))
    future_connections['adamic_adar'] = pd.Series([z for x, y, z in nx.adamic_adar_index(G)])
    #     L.sort(key=operator.itemgetter(2), reverse=True)
    #     print(L)

    # Method 5:
    # Preferential Attachment
    # In the preferential attachment model, nodes with high degree get more neighbors
    # degree_source * degree_target
    future_connections['pref_att'] = pd.Series([z for x, y, z in nx.preferential_attachment(G)])
    #     print(L)

    # Measure 6:
    # Community Common Neighbors
    # Number of common neighbors with bonus of 1 for each neighbor in same community
    # f(u) = 1 if same community else 0
    # sum(f(u) * degree)
    for i, dept in enumerate(nx.get_node_attributes(G, 'Department')):
        G.node[i]['community'] = dept
    future_connections['com_common_nb'] = pd.Series([z for x, y, z in nx.cn_soundarajan_hopcroft(G)])
    #     L.sort(key=operator.itemgetter(2), reverse=True)
    #     print(L)

    # Measure 7:
    # Community Resource Allocation
    # Similar to resource allocation index, but only considering nodes in the same community
    # f(u) = 1 if same community else 0
    # sum(f(u)/degree)
    future_connections['com_resource'] = pd.Series([z for x, y, z in nx.ra_index_soundarajan_hopcroft(G)])
    #     L.sort(key=operator.itemgetter(2), reverse=True)
    #     print(L)

    print(df.head())

    #     #
    #     df_fc_train = future_connections.loc[~df_fc_test_mask, :]
    #     df_fc_test = future_connections.loc[df_fc_test_mask, :]
    #     y_train = df_fc_train.loc[:, 'Future Connection']
    #     y_test = df_fc_test.loc[:, 'Future Connection']
    #     X_train = df_fc_train.index
    #     X_test = df_fc_test.index

    #     def auc_scores(model, *args, k=5, threshold=0.50):
    #         """CV scores"""
    #         X, y = args
    #         predictions = cross_val_predict(model, X, y, cv=k, n_jobs=-1)
    #         print('AUC - Test predict  {:.2%}'.format(roc_auc_score(y, predictions)))

    #     classifiers = [
    # #         GaussianNB(),
    # #         DecisionTreeClassifier(random_state=0),
    # #         DecisionTreeClassifier(max_depth=3, random_state=0),
    # #         DecisionTreeClassifier(max_depth=4, random_state=0),
    # #         DecisionTreeClassifier(max_depth=5, random_state=0),
    # #         DecisionTreeClassifier(max_depth=6, random_state=0),
    #         GradientBoostingClassifier(random_state=0),
    # #         GradientBoostingClassifier(learning_rate=0.08, random_state=0),
    # #         GradientBoostingClassifier(learning_rate=0.12, random_state=0),
    # #         GradientBoostingClassifier(learning_rate=0.1, max_depth=3, random_state=0),
    # #         GradientBoostingClassifier(learning_rate=0.1, max_depth=4, random_state=0),
    # #         RandomForestClassifier(n_estimators=100, random_state=0),
    # #         AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=0),
    # #         KNeighborsClassifier(),
    # #         KNeighborsClassifier(n_neighbors=4),
    # #         LinearSVC(random_state=0)
    #         ]

    #     for model in classifiers:
    # #         print('-'*80)
    # #         print(model)

    #         # Training scores
    # #         clf_train = model.fit(X_train, y_train)
    # #         pred_train = clf_train.predict(X_train)
    # #         print('AUC - Train pred    {:.2%}'.format(roc_auc_score(y_train, pred_train)))

    #         # CV scores
    #         clf = model.fit(X_train, y_train)
    # #         auc_scores(clf, X_train, y_train)

    #     # Predict
    #     predicted = clf.predict(X_test)
    #     pred_series = pd.Series(predicted)
    #     assert type(pred_series) == pd.Series, 'wtf: ' + str(type(pred_series))

    return pred_series

# ---------------------------------------------> last 100% <-------------------------------------------- #

import networkx as nx
import pandas as pd
import numpy as np
import pickle


P1_Graphs = pickle.load(open('A4_graphs','rb'))
P1_Graphs

def graph_identification():
    #     import matplotlib as mpl
    #     import matplotlib.pyplot as plt

    # Your Code Here
    algs = ['PA', 'SW_L', 'SW_H']

    # Real World: degree dist is Power Law
    # PA: Power Law; log/log degree distribution is straight line
    # => max clustering = 0.1, max shortest path = 7,5

    # Small world: degree dist is not Power Law
    # More nodes: higher Avg shortest path, Avg lower clustering
    # Higher rewiring p: lower clustering and lower shortest path
    # - SW_L: Lattice - Small World; higher shortest paths > 7.5, higher clustering
    # - SW_H: Small world - Random;  lower shortest paths < 7.5, lower clustering
    # SW_L: => max clustering = 0.1, max shortest path = 7,5
    # SW_H: => max clustering = 0.02, max shortest path = 4,5

    def graph_algorithm_params(G):

        degrees = G.degree()
        #         print(len(degrees))
        degree_set = sorted(set(degrees.values()))
        degree_list = list(degrees.values())
        histogram = [degree_list.count(i) / float(nx.number_of_nodes(G)) for i in degree_set]
        print(nx.average_clustering(G), nx.average_shortest_path_length(G), len(histogram))
        if histogram[0] > histogram[1] > histogram[2] > histogram[3]:
            print('PA')
        elif (nx.average_shortest_path_length(G) < 7.5) | (nx.average_clustering(G) < 0.25):
            print('SW_H')
        else:
            print('SW_L')
        plt.bar(degree_set, histogram)
        plt.xlabel('Degree')
        plt.ylabel('Fraction of Nodes')
        plt.show()

    #     graph_algorithm_params(nx.barabasi_albert_graph(1000, 1))
    #     graph_algorithm_params(nx.barabasi_albert_graph(1000, 2))
    #     graph_algorithm_params(nx.barabasi_albert_graph(1000, 5))
    #     graph_algorithm_params(nx.barabasi_albert_graph(1000, 20))

    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 2, 0.1, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 5, 0.1, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 10, 0.1, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 2, 0.3, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 6, 0.25, 20))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 2, 0.7, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 5, 0.7, 10))
    #     graph_algorithm_params(nx.connected_watts_strogatz_graph(1000, 12, 1, 10))

    #     for graph in P1_Graphs:
    #         graph_algorithm_params(graph)

    algs = ['PA', 'SW_L', 'SW_L', 'PA', 'SW_H']

    return algs


graph_identification()

G = nx.read_gpickle('email_prediction.txt')
print(nx.info(G))
G.nodes(data=True)

def salary_predictions():
    # Import preprocessing, selection and metrics
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, GridSearchCV
    from sklearn.metrics import roc_auc_score
    from sklearn.dummy import DummyClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neural_network import MLPClassifier

    # Your Code Here
    G.edges(data=True)

    df = pd.DataFrame(index=G.nodes())
    df['Department'] = pd.Series(nx.get_node_attributes(G, 'Department'))
    df['ManagementSalary'] = pd.Series(nx.get_node_attributes(G, 'ManagementSalary'))
    df['clustering'] = pd.Series(nx.clustering(G))
    df['degree'] = pd.Series(G.degree())
    df['degree_cent'] = pd.Series(nx.degree_centrality(G))
    df['closeness'] = pd.Series(nx.closeness_centrality(G, normalized=True))
    df['betweenness'] = pd.Series(nx.betweenness_centrality(G, normalized=True))
    df['pagerank'] = pd.Series(nx.pagerank(G, alpha=0.80))
    df['hub'] = pd.Series(nx.hits(G)[0])
    df['authority'] = pd.Series(nx.hits(G)[1])

    df_test_mask = pd.isnull(df.loc[:, 'ManagementSalary'])
    df_train = df.loc[~df_test_mask, :]
    df_test = df.loc[df_test_mask, :]

    #
    y_train = df_train.pop('ManagementSalary')
    X_train = df_train
    df_test.drop('ManagementSalary', axis=1, inplace=True)
    X_test = df_test
    idx_test = df_test.index

    #
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    def auc_scores(model, *args, k=5, threshold=0.50):
        """CV scores"""
        X, y = args
        predictions = cross_val_predict(model, X, y, cv=k, n_jobs=-1)
        pred_probas = (cross_val_predict(model, X, y, cv=k, method='predict_proba', n_jobs=-1)[:, 1] > threshold) * 1
        print('AUC - Test predict  {:.2%}'.format(roc_auc_score(y, predictions)))
        print('AUC - Test probabil {:.2%}'.format(roc_auc_score(y, pred_probas)))

    classifiers = [
        #         GaussianNB(),
        #         DecisionTreeClassifier(random_state=0),
        #         DecisionTreeClassifier(max_depth=3, random_state=0),
        #         DecisionTreeClassifier(max_depth=4, random_state=0),
        #         DecisionTreeClassifier(max_depth=5, random_state=0),
        #         DecisionTreeClassifier(max_depth=6, random_state=0),
        GradientBoostingClassifier(random_state=0),
        #         GradientBoostingClassifier(learning_rate=0.08, random_state=0),
        #         GradientBoostingClassifier(learning_rate=0.12, random_state=0),
        #         GradientBoostingClassifier(learning_rate=0.1, max_depth=3, random_state=0),
        #         GradientBoostingClassifier(learning_rate=0.1, max_depth=4, random_state=0),
        #         RandomForestClassifier(n_estimators=100, random_state=0),
        #         AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=0),
        #         KNeighborsClassifier(),
        #         KNeighborsClassifier(n_neighbors=4),
        #         MLPClassifier(hidden_layer_sizes = [10, 5], alpha = 5,
        #                        random_state = 0, solver='lbfgs', verbose=0)
        #         LinearSVC(random_state=0)
    ]

    for model in classifiers:
        #         print('-'*80)
        #         print(model)

        # Training scores
        #         clf_train = model.fit(X_train, y_train)
        #         pred_train = clf_train.predict(X_train)
        #         print('AUC - Train pred    {:.2%}'.format(roc_auc_score(y_train, pred_train)))

        # CV scores
        clf = model.fit(X_train, y_train)
    #         auc_scores(clf, X_train, y_train)

    # Predict
    #     predicted = [x for x, y in clf.predict_proba(X_test)]
    predicted = pd.DataFrame(clf.predict_proba(X_test), columns=clf.classes_)
    predicted['idx'] = idx_test
    predicted.set_index('idx', inplace=True)
    predicted.drop(0.0, axis=1, inplace=True)
    pred_series = predicted.loc[:, 1.0]  # pd.Series(predicted.values)
    assert type(pred_series) == pd.Series, 'wtf: ' + str(type(pred_series))

    return pred_series


salary_predictions()

future_connections = pd.read_csv('Future_Connections.csv', index_col=0, converters={0: eval})
future_connections.head(10)

new_connections_predictions()

#
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)