use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use ordered_float::OrderedFloat;
use rand::Rng;
use crate::vector::{DistanceCalculator, VectorItem};
use crate::node::Node;

pub struct HnswIndex {
    nodes: Arc<Mutex<HashMap<usize, Node>>>,
    max_elements: usize,
    level_lambda: f64,
    max_level: usize,
    distance_calculator: Box<dyn DistanceCalculator>,
}

impl HnswIndex {
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

    pub fn add(&self, item: VectorItem) -> Result<(), String> {
        let mut nodes = self.nodes.lock().unwrap();
        let node_id = item.id;
        let layer = self.random_layer();
        let new_node = Node {
            id: node_id,
            connections: vec![Vec::new(); self.max_level + 1],
            item: item.clone(),
            layer,
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
            top_k_items.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            if top_k_items.len() > k {
                top_k_items.pop();
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
        let entry_point = nodes.keys().cloned().next().ok_or("No nodes in the graph")?;
        let mut candidates: HashMap<usize, f64> = HashMap::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut closest_distance: f64;

        for layer in (0..=self.max_level).rev() {
            let mut current_node = entry_point;
            loop {
                let mut closest_node = None;
                closest_distance = f64::MAX;
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
                    break;
                }
            }
    
            candidates.insert(current_node, closest_distance);
        }
    
        let top_k_items = candidates
            .iter()
            .take(k)
            .map(|(&id, &_dist)| VectorItem {
                id,
                vector: nodes.get(&id).unwrap().item.vector.clone(),
            })
            .collect::<Vec<VectorItem>>();
    
        Ok(top_k_items)
    }    
}
