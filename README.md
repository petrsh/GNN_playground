## GNN playground

This repository serves as a playground for gaining hands-on experience with graph neural networks, specifically in context of single-cell multiomics, using [PyTorch Geometric library](https://pytorch-geometric.readthedocs.io/en/latest/index.html). The repository currently contains my initial simple try to build GNN model to predict CITE from GEX and GEX from ATAC using dataset of CD34+ hematopoietic stem and progenitor cells  from four human donors at five time points from [Kaggle Open Problems in Single-Cell Analysis](https://www.kaggle.com/competitions/open-problems-multimodal/) competition. 

GNN allows us to utilize deep learning on graphs. In a nutshell, in a GNN, information is propagated through the edges and nodes via a series of iterative message-passing steps which represents generalization of the convolution operator to graphs. During each iteration, nodes exchange information with their neighbors. Subsequently, an aggregation step consolidates these messages, allowing each node to integrate information from its local neighborhood. This iterative process is repeated over multiple steps, enabling the model to capture complex patterns and dependencies within the graph. There are two main types of GNN variations - homogenous and heterogenous:  

In a homogeneous GNN, all nodes and edges are of the same type. When applied in the context of single-cell analysis, this involves creating a cell-to-cell graph, such as a graph of k nearest neighbors where nearest is quantified by a similarity metric.

In a heterogeneous GNN, nodes and edges can be of different types. In the context of single-cell analysis, a gene-to-cell graph is created where weighted edges between gene and cell nodes represent gene counts. One of the benefits of using a heterogeneous architecture is that it allows the incorporation of priors, for example, in the form of edges between interacting genes or genes from the same pathways.
Layers

GNN layers have different strengths and biases built into them (e.g. GCN is good at smoothing the surrounding neighborhood into a single embedding while GAT can learn more fine-granular patterns, e.g. to discard certain neighbors) and generally cannot be used in all situations. E.g., for heterogenous architecture we can only use layers supporting bipartite message passing, i.e., we cannot use GCNConv that can only be used for passing messages to the same node type. For example, GraphConv is a layer that supports bipartite message passing and edge weights and is good at capturing higher-order interactions i.e., it considers relationships between more nodes that have some shared characteristic.

Useful [cheatsheet](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html) for choosing a suitable layer.

### Aggregation
GNN aggregation is pivotal for consolidating information gathered from neighboring nodes. Various aggregation functions, including summation or mean, determine how a node combines information from its neighbors. Additionally, attention mechanisms can be leveraged to upweight neighboring nodes, allowing the model to focus on more relevant information. The choice of the appropriate aggregation method can significantly impact the model's performance.

(A principled approach to aggregations)[https://medium.com/@pytorch_geometric/a-principled-approach-to-aggregations-983c086b10b3] 

### Mini-batching
Mini-batching is a technique employed in DL to enhance efficiency, reduce memory requirements, and facilitate scalability. For heterogenous GNN, we can use [hetogenous mini-batch graph sampling](https://arxiv.org/pdf/2003.01332.pdf). 

(Useful discussion)[https://github.com/pyg-team/pytorch_geometric/discussions/6707] on num_samples and batch_size parameters.

### Interesting applications of GNN in single-cell

GNNs have numerous use cases in biology and single-cell data analysis including imputation, clustering, dimensionality reduction, cell annotation, gene regulatory network inference, cell-cell communication and other applications such as drug target nomination. Here is a collection of interesting application that I’ve stumbled upon:

-	overview - Graph representation learning for single-cell biology https://www.sciencedirect.com/science/article/pii/S2452310021000329#bib56
-	Graph Neural Networks for Multimodal Single-Cell Data Integration (https://arxiv.org/abs/2203.01884) - https://github.com/OmicsML/dance - this method won modality prediction in NeurIPS 2021 competition

-	Contextualizing protein representations using deep learning on protein networks and single-cell data – https://www.biorxiv.org/content/10.1101/2023.07.18.549602v1 This paper presents a flexible geometric DL approach trained on contextualized protein interaction networks to generate context-aware protein representations. The model can be used to nominate contextualized drug targets.
	
-	scGNN: scRNA-seq Dropout Imputation via Induced Hierarchical Cell Similarity Graph (https://arxiv.org/pdf/2008.03322.pdf)
-	Graining Insight into SARS-Cov-2 Infection and COVID-19 Severity Using Self-Supervised Edge Features and Graph Neural Network (https://arxiv.org/abs/2006.12971)
-	Cellograph: A Semi-supervised Approach to Analyzing Multi-condition Single-cell RNA Sequencing Data Using Graph Neural Networks (https://www.biorxiv.org/content/10.1101/2023.02.24.528672v1) – uses GNN to quantify the effects of perturbations in single cells

(Machine learning with graphs course)[https://web.stanford.edu/class/cs224w/] 
