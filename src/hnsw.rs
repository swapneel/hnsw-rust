use crate::vector::{DistanceCalculator, VectorItem};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex};
use rand::Rng;


const M: usize = 16;
const M_MAX0: usize = 32;
const EF_CONSTRUCTION: usize = 100;
const EF_SEARCH: usize = 64;

#[derive(Clone, Debug)]
struct Neighbor {
    id: usize,
    distance: f64,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub connections: Vec<Vec<usize>>,
    pub item: VectorItem,
    pub layer: usize,
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .reverse()
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

pub struct HnswIndex {
    nodes: Arc<Mutex<HashMap<usize, Node>>>,
    entry_point: Arc<Mutex<Option<usize>>>,
    level_lambda: f64,
    max_level: usize,
    distance_calculator: Box<dyn DistanceCalculator + Send + Sync>,
}

impl HnswIndex {
    pub fn new(
        distance_calculator: Box<dyn DistanceCalculator + Send + Sync>,
    ) -> Self {
        HnswIndex {
            nodes: Arc::new(Mutex::new(HashMap::new())),
            entry_point: Arc::new(Mutex::new(None)),
            level_lambda: 1.0 / (M as f64).ln(),
            max_level: 16,  // Default max level
            distance_calculator,
        }
    }

    pub fn add(&self, item: VectorItem) -> Result<(), String> {
        let node_id = item.id;
        let node_level = self.random_level();
        let mut connections = vec![Vec::with_capacity(if node_level == 0 { M_MAX0 } else { M }); node_level + 1];
    
        let mut nodes = self.nodes.lock().unwrap();
        let mut entry_point = self.entry_point.lock().unwrap();

        // Handle first node case
        if nodes.is_empty() {
            let new_node = Node {
                id: node_id,
                connections: vec![Vec::with_capacity(M_MAX0); node_level + 1],
                item: item.clone(),
                layer: node_level,
            };
            nodes.insert(node_id, new_node);
            *entry_point = Some(node_id);
            return Ok(());
        }

        // Find entry point for insertion
        let curr_ep = entry_point.unwrap();
        let mut curr_dist = self.calculate_distances(&item, &nodes[&curr_ep].item);

        // Insert at each layer
        for level in (0..=node_level).rev() {
            let neighbors = self.search_at_layer(&nodes, curr_ep, &item, level, 
                if level == 0 { EF_CONSTRUCTION } else { M })?;
            
            let selected = self.select_neighbors(&nodes, &item, &neighbors, level)?;
            
            // Create new node's connections at this level
            if level < connections.len() {
                connections[level] = selected.clone();
            }

            // Update reverse connections
            for &neighbor_id in &selected {
                // Clone the nodes we need to avoid borrow conflicts
                let neighbor_item = nodes[&neighbor_id].item.clone();
                let neighbor_dist = self.calculate_distances(&item, &neighbor_item);
                let neighbor_level = nodes[&neighbor_id].layer;
                
                let reverse_selected = self.select_neighbors(
                    &nodes,
                    &neighbor_item,
                    &[Neighbor { id: node_id, distance: neighbor_dist }],
                    level
                )?;
                
                // Now do the mutable update
                if let Some(neighbor_node) = nodes.get_mut(&neighbor_id) {
                    if level < neighbor_node.connections.len() {
                        neighbor_node.connections[level] = reverse_selected;
                    }
                }
            }            
        }

        // Insert the new node
        let new_node = Node {
            id: node_id,
            connections,
            item,
            layer: node_level,
        };
        nodes.insert(node_id, new_node);

        // Update entry point if necessary
        if node_level > nodes[&entry_point.unwrap()].layer {
            *entry_point = Some(node_id);
        }

        Ok(())
    }

    fn select_neighbors(
        &self,
        nodes: &HashMap<usize, Node>,
        query: &VectorItem,
        candidates: &[Neighbor],
        level: usize,
    ) -> Result<Vec<usize>, String> {
        let max_connections = if level == 0 { M_MAX0 } else { M };
        let mut selected = Vec::with_capacity(max_connections);
        let mut remaining: Vec<_> = candidates.to_vec();
        
        remaining.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        for candidate in remaining.iter().take(max_connections) {
            let mut should_add = true;
            for &existing in &selected {
                let dist_between = self.calculate_distances(
                    &nodes[&candidate.id].item,
                    &nodes[&existing].item
                );
                
                if dist_between < candidate.distance {
                    should_add = false;
                    break;
                }
            }
            
            if should_add {
                selected.push(candidate.id);
            }
        }
        
        Ok(selected)
    }


    fn _insert_at_layer(
        &self,
        nodes: &mut HashMap<usize, Node>,
        current_id: usize,
        query: &VectorItem,
        level: usize,
        ef: usize,
    ) -> Result<Vec<usize>, String> {
        let neighbors = self.search_at_layer(nodes, current_id, query, level, ef)?;
        let selected = self.select_neighbors(nodes, query, &neighbors, level)?;
        
        // Update reverse connections
        for &neighbor_id in &selected {
            let neighbor_dist = self.calculate_distances(query, &nodes[&neighbor_id].item);
            let candidate = Neighbor {
                id: current_id,
                distance: neighbor_dist,
            };
            
            let reverse_conns = self.select_neighbors(
                nodes,
                &nodes[&neighbor_id].item,
                &[candidate],
                level
            )?;
            
            if let Some(node) = nodes.get_mut(&neighbor_id) {
                if level < node.connections.len() {
                    node.connections[level] = reverse_conns;
                }
            }
        }
        
        Ok(selected)
    }

    fn _select_neighbors_heuristic(
        &self,
        nodes: &HashMap<usize, Node>,
        _query: &VectorItem,
        candidates: &[Neighbor],
        level: usize,
    ) -> Result<Vec<usize>, String> {
        let max_connections = if level == 0 { M_MAX0 } else { M };
        let mut selected = Vec::with_capacity(max_connections);
        
        for candidate in candidates.iter().take(max_connections) {
            let mut should_add = true;
            for &existing in &selected {
                let dist_between = self.calculate_distances(
                    &nodes[&candidate.id].item,
                    &nodes[&existing].item
                );
                
                if dist_between < candidate.distance {
                    should_add = false;
                    break;
                }
            }
            
            if should_add {
                selected.push(candidate.id);
            }
        }
        
        Ok(selected)
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut level = 0;
        while rng.gen::<f64>() < self.level_lambda && level < self.max_level {
            level += 1;
        }
        level
    }

    fn calculate_distances(&self, item1: &VectorItem, item2: &VectorItem) -> f64 {
        self.distance_calculator.calculate(item1, item2)
    }

    fn search_at_layer(
        &self,
        nodes: &HashMap<usize, Node>,
        entry_point: usize,
        query: &VectorItem,
        level: usize,
        ef: usize,
    ) -> Result<Vec<Neighbor>, String> {
        let entry_node = nodes.get(&entry_point)
            .ok_or_else(|| format!("Entry point {} not found", entry_point))?;
    
        if level >= entry_node.connections.len() {
            return Ok(Vec::new());
        }
    
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();
    
        let initial_dist = self.calculate_distances(query, &entry_node.item);
        let initial = Neighbor {
            id: entry_point,
            distance: initial_dist,
        };
    
        candidates.push(initial.clone());
        results.push(initial);
        visited.insert(entry_point);
    
        while let Some(current) = candidates.pop() {
            // Get worst distance in results
            let furthest_dist = results.peek().map_or(f64::INFINITY, |n| n.distance);
    
            if current.distance > furthest_dist {
                break;
            }
    
            if let Some(node) = nodes.get(&current.id) {
                if level < node.connections.len() {
                    for &neighbor_id in &node.connections[level] {
                        if visited.insert(neighbor_id) {
                            if let Some(neighbor_node) = nodes.get(&neighbor_id) {
                                let distance = self.calculate_distances(query, &neighbor_node.item);
                                let neighbor = Neighbor {
                                    id: neighbor_id,
                                    distance,
                                };
    
                                if results.len() < ef || distance < furthest_dist {
                                    candidates.push(neighbor.clone());
                                    results.push(neighbor);
                                    
                                    if results.len() > ef {
                                        results.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    
        Ok(results.into_sorted_vec())
    }

    fn _select_connections_for_level(
        &self,
        nodes: &HashMap<usize, Node>,
        _item: &VectorItem,
        candidates: &[Neighbor],
        level: usize,
    ) -> Vec<usize> {
        let max_connections = if level == 0 { M_MAX0 } else { M };
        let mut selected = Vec::with_capacity(max_connections);
        let mut remaining: Vec<_> = candidates.to_vec();
        remaining.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        'outer: while selected.len() < max_connections && !remaining.is_empty() {
            let current = remaining.remove(0);

            // Check if this connection would be closer to any already selected neighbor
            for &selected_id in &selected {
                let selected_node = &nodes[&selected_id];
                let dist_between =
                    self.calculate_distances(&nodes[&current.id].item, &selected_node.item);
                if dist_between < current.distance {
                    continue 'outer;
                }
            }

            selected.push(current.id);
        }

        selected
    }

    pub fn search(&self, query: &VectorItem, k: usize) -> Result<Vec<VectorItem>, String> {
        let nodes = self.nodes.lock().unwrap();
        let entry_point = self.entry_point.lock().unwrap();
    
        if nodes.is_empty() {
            return Ok(Vec::new());
        }
    
        let ep = entry_point.unwrap();
        let mut curr_ep = ep;
        let mut curr_dist = self.calculate_distances(query, &nodes[&curr_ep].item);
        let ep_level = nodes[&ep].layer;
    
        // First traverse down to find a good entering point
        for level in (1..=ep_level).rev() {
            loop {
                let mut best_dist = curr_dist;
                let mut best_ep = curr_ep;
                
                // Check all neighbors at this level
                if let Some(node) = nodes.get(&curr_ep) {
                    if level < node.connections.len() {
                        for &neighbor_id in &node.connections[level] {
                            let dist = self.calculate_distances(query, &nodes[&neighbor_id].item);
                            if dist < best_dist {
                                best_dist = dist;
                                best_ep = neighbor_id;
                            }
                        }
                    }
                }
                
                if best_ep == curr_ep {
                    break;  // No better neighbor found
                }
                curr_ep = best_ep;
                curr_dist = best_dist;
            }
        }
    
        // Perform final search at layer 0 with larger ef
        let mut neighbors = self.search_at_layer(&nodes, curr_ep, query, 0, EF_SEARCH)?;
        
        // Sort by distance before returning
        neighbors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        Ok(neighbors
            .into_iter()
            .take(k)
            .map(|n| nodes[&n.id].item.clone())
            .collect())
    }
    
    pub fn batch_add(&self, items: Vec<VectorItem>) -> Result<(), String> {
        for item in items {
            self.add(item)?;
        }
        Ok(())
    }

    pub fn get_stats(&self) -> IndexStats {
        let nodes = self.nodes.lock().unwrap();
        let mut level_counts = HashMap::new();
        let mut total_connections = 0;

        for node in nodes.values() {
            *level_counts.entry(node.layer).or_insert(0) += 1;
            total_connections += node
                .connections
                .iter()
                .map(|conns| conns.len())
                .sum::<usize>();
        }

        IndexStats {
            total_nodes: nodes.len(),
            level_distribution: level_counts,
            total_connections,
            max_level: self.max_level,
        }
    }
}

#[derive(Debug)]
pub struct IndexStats {
    pub total_nodes: usize,
    pub level_distribution: HashMap<usize, usize>,
    pub total_connections: usize,
    pub max_level: usize,
}

#[cfg(test)]
mod tests {
    use crate::EuclideanDistance;

    use super::*;
    use rand::distributions::{Distribution, Uniform};
    use rand::thread_rng;

    fn generate_random_vector(dim: usize) -> Vec<f64> {
        let mut rng = thread_rng();
        let dist = Uniform::from(-1.0..1.0);
        (0..dim).map(|_| dist.sample(&mut rng)).collect()
    }

    #[test]
    fn test_basic_insertion() {
        let index = HnswIndex::new(Box::new(EuclideanDistance));
        let item = VectorItem {
            id: 1,
            vector: vec![1.0, 0.0],
        };
        assert!(index.add(item).is_ok());
    }
    
    #[test]
    fn test_batch_add() {
        let index = HnswIndex::new(Box::new(EuclideanDistance));
        let mut items = Vec::new();

        for i in 0..100 {
            items.push(VectorItem {
                id: i,
                vector: generate_random_vector(10),
            });
        }

        assert!(index.batch_add(items).is_ok());
        let stats = index.get_stats();
        assert_eq!(stats.total_nodes, 100);
    }
}
