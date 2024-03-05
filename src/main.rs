use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use ordered_float::OrderedFloat;
use rand::Rng;

#[derive(Clone, Debug)]
struct VectorItem {
    id: usize,
    vector: Vec<f64>,
}

trait DistanceCalculator {
    fn calculate(&self, item1: &VectorItem, item2: &VectorItem) -> f64;
}

// A simple Euclidean distance calculator as an example
struct EuclideanDistance;

impl DistanceCalculator for EuclideanDistance {
    fn calculate(&self, item1: &VectorItem, item2: &VectorItem) -> f64 {
        item1.vector.iter().zip(item2.vector.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

struct Node {
    id: usize,
    connections: Vec<Vec<usize>>,
    item: VectorItem,
    layer: usize,
}

struct HnswIndex {
    nodes: Arc<Mutex<HashMap<usize, Node>>>,
    max_elements: usize,
    level_lambda: f64,
    max_level: usize,
    distance_calculator: Box<dyn DistanceCalculator>,
}

impl HnswIndex {
    // Initialization of HNSW index
    pub fn new(max_elements: usize, level_lambda: f64, max_level: usize, distance_calculator: Box<dyn DistanceCalculator>) -> Self {
        HnswIndex {
            nodes: Arc::new(Mutex::new(HashMap::new())),
            max_elements,
            level_lambda,
            max_level,
            distance_calculator,
        }
    }

    fn random_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut layer = 0;
        while rng.gen::<f64>() < self.level_lambda && layer < self.max_level {
            layer += 1;
        }
        layer
    }


    // // Add method
    // pub fn add(&self, item: VectorItem) -> Result<(), String> {
    //     let mut nodes = self.nodes.lock().unwrap();
    //     let node_id = item.id;
    //     let new_node = Node {
    //         id: node_id,
    //         connections: vec![Vec::new(); self.max_level + 1],
    //         item: item.clone(),
    //     };
    //     nodes.insert(node_id, new_node);
    //     Ok(())
    // }

    pub fn add(&self, item: VectorItem) -> Result<(), String> {
        let mut nodes = self.nodes.lock().unwrap();
        let node_id = item.id;
        let layer = self.random_layer(); // Determine the layer of the node
        let new_node = Node {
            id: node_id,
            connections: vec![Vec::new(); self.max_level + 1],
            item: item.clone(),
            layer, // Set the layer
        };
        nodes.insert(node_id, new_node);
        // Logic for connecting the node in the graph goes here
        Ok(())
    }

    pub fn search(&self, query: &VectorItem, k: usize) -> Result<Vec<VectorItem>, String> {
        let nodes = self.nodes.lock().unwrap();
        let mut top_k_items: Vec<(OrderedFloat<f64>, VectorItem)> = Vec::new();

        for node in nodes.values() {
            let dist = OrderedFloat::<f64>(self.distance_calculator.calculate(query, &node.item));
            top_k_items.push((-dist, node.item.clone()));
            top_k_items.sort_unstable_by(|a, b| b.0.cmp(&a.0)); // Sort in descending order of distance
            if top_k_items.len() > k {
                top_k_items.pop(); // Keep only the top k items
            }
        }

        let mut result = Vec::new();
        for (_, item) in top_k_items.into_iter() {
            result.push(item);
        }
        Ok(result)
    }

    pub fn search_greedy(&self, query: &VectorItem, k: usize) -> Result<Vec<VectorItem>, String> {

        let nodes = self.nodes.lock().unwrap();
        // Dummy entry point: for example, we can just pick the first node's ID if it exists
        let entry_point = nodes.keys().cloned().next().ok_or("No nodes in the graph")?;
        // Candidates: Using a HashMap to store node IDs and their distance from the query
        let mut candidates: HashMap<usize, f64> = HashMap::new();
        // Visited: Using a HashSet to keep track of visited node IDs
        let mut visited: HashSet<usize> = HashSet::new();
        let mut closest_distance: f64;


        for layer in (0..=self.max_level).rev() {
            let mut current_node = entry_point;
            loop {
                let mut closest_node = None;
                closest_distance = f64::MAX;
                // Perform greedy search in the current layer
                for &neighbor_id in &nodes[&current_node].connections[layer] {
                    if visited.insert(neighbor_id) {
                        let neighbor = &nodes[&neighbor_id].item;
                        let distance = self.distance_calculator.calculate(query, neighbor);
                        if distance < closest_distance {
                            closest_node = Some(neighbor_id);
                            closest_distance = distance;
                        }
                    }
                }
    
                if let Some(closest) = closest_node {
                    current_node = closest;
                } else {
                    break; // No closer node found, move to the lower layer
                }
            }
    
            // Add current node to candidates
            candidates.insert(current_node, closest_distance);
        }
    
        // Refine candidates on the bottom layer to find top k
        let mut top_k_items = candidates
            .iter()
            .take(k)
            .map(|(&id, &dist)| VectorItem {
                id,
                vector: nodes.get(&id).unwrap().item.vector.clone(),
            })
            .collect::<Vec<VectorItem>>();
    
        Ok(top_k_items)
    }    
}



// fn main() {
//     let hnsw = HnswIndex::new(10000, 1.0 / 3.0, 16, Box::new(EuclideanDistance));

//     // Example: Add items
//     let item1 = VectorItem { id: 1, vector: vec![1.0, 2.0] };
//     let item2 = VectorItem { id: 2, vector: vec![2.0, 3.0] };
//     hnsw.add(item1).unwrap();
//     hnsw.add(item2).unwrap();

//     // Example: Search
//     let query = VectorItem { id: 3, vector: vec![1.5, 2.5] };
//     let results = hnsw.search(&query, 2).unwrap();
//     for result in results {
//         println!("Found item with id: {}", result.id);
//     }
// }

fn main() {
    let hnsw = HnswIndex::new(10000, 1.0 / 3.0, 16, Box::new(EuclideanDistance));

    // Add a larger number of items
    for i in 0..1000 {
        let mut rng = rand::thread_rng();
        let vector: Vec<f64> = (0..10).map(|_| rng.gen_range(0.0..10.0)).collect();
        let item = VectorItem { id: i, vector };
        hnsw.add(item).unwrap();
    }

    // Test search with different queries
    let test_queries = vec![
        vec![1.0, 2.0],
        vec![5.0, 5.0],
        vec![9.0, 0.0],
    ];

    for query_vec in test_queries {
        let query = VectorItem { id: 9999, vector: query_vec };
        match hnsw.search(&query, 5) {
            Ok(results) => {
                println!("Search results for query {:?}:", query.vector);
                for result in results {
                    println!("Found item with id: {}", result.id);
                }
            },
            Err(e) => println!("Search error: {}", e),
        }
    }

}
