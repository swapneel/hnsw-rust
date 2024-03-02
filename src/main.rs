use std::collections::{HashMap};
use std::sync::{Arc, Mutex};
use ordered_float::OrderedFloat;
use rand::Rng;

// Assuming a simple vector-based item for demonstration
#[derive(Clone, Debug)]
struct VectorItem {
    id: usize,
    vector: Vec<f64>,
}

// Define a trait for distance calculation
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

// Node structure
struct Node {
    id: usize,
    connections: Vec<Vec<usize>>,
    item: VectorItem,
    layer: usize,
}

// HNSW index structure
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

    // Optionally, include performance metrics
}
