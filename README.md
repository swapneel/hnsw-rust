## hnsw-rust - a fast HNSW implementation in Rust

### Technical Details
hnsw-rust is a Rust implementation of The Hierarchical Navigable Small World (HNSW) algorithm. HNSW is a notable advancement in Approximate Nearest Neighbor (ANN) search in high-dimensional spaces, fundamentally altering our approach to these problems. The algorithm constructs a layered graph structure, where higher layers (less dense) are used for rapid global navigation, while lower layers (more dense) facilitate fine-grained local search. This structure mirrors the 'small world' phenomenon observed in social networks, where short path lengths exist between any two nodes (Watts and Strogatz, 1998).

HNSW's search efficiency arises from its unique use of a greedy heuristic. It commences from a high layer and iteratively moves to the node closest to the target, transitioning down to denser layers until the nearest neighbors are refined. This method of layer traversal for nearest neighbor search finds its roots in earlier works like Kleinberg's small-world model, which also highlights efficient navigation in sparse, high-dimensional spaces (Kleinberg, 2000).

<img width="314" alt="image" src="https://github.com/swapneel/hnsw-rust/assets/6643641/72a47b3d-7a7b-49b6-836c-70e2bc6efa98">

New nodes are inserted starting from the lowest layer, with their inclusion in each subsequent higher layer governed by a probabilistic threshold. This strategy is influenced by earlier research in dynamic random graphs and scale-free networks, which also deal with node connections based on probabilistic models (Barabási and Albert, 1999). The probability of ascending to higher layers decreases exponentially, a method validated by research emphasizing the balance between exploration and exploitation in search algorithms (Arya et al., 1998).


### References:
Malkov, Yu. A., & Yashunin, D. A. (2016). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. arXiv preprint arXiv:1603.09320.
Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of ‘small-world’ networks. Nature, 393(6684), 440-442.
Kleinberg, J. M. (2000). The small-world phenomenon: An algorithmic perspective. Proceedings of the 32nd annual ACM symposium on Theory of computing.
Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks. Science, 286(5439), 509-512.
Arya, S., Mount, D. M., Netanyahu, N. S., Silverman, R., & Wu, A. Y. (1998). An optimal algorithm for approximate nearest neighbor searching fixed dimensions. Journal of the ACM (JACM), 45(6), 891-923.
Aumüller, M., Bernhardsson, E., & Faithfull, A. (2017). ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms. Information Systems, arXiv:1807.05614.
