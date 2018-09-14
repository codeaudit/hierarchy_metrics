# hierarchy_metrics

This repository contains the reference implementation in Python of the relative entropy, a metric for assessing the quality of the hierachical clustering of a graph, as described in:

[Learning Graph Representations by Dendrograms](http://arxiv.org/abs/1807.05087), 2018

## Dependency

The implementation depends on the `networkx` package,
which can be installed using `pip`.

```python
sudo pip install networkx
```

## Getting started

```python
from hierarchy_metrics import *
```

Hierarchy of the Karate Club graph using Newman's algorithm:

```python
graph = nx.karate_club_graph()
dendrogram = hierarchical_clustering(graph, algorithm = "newman")
```

Metrics:
 
```python
print("Quality:", relative_entropy(graph, dendrogram))
print("Cost:", dasgupta_cost(graph, dendrogram))
```
Quality: 1.3526175451991203  
Cost: 0.36143984220907294

## Experiments

Experiments on real and synthetic data are available as a Jupyter notebook:

```python
experiments.ipynb
```

## License

Released under the 3-clause BSD license.

