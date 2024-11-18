use hnsw_rust::{HnswIndex, VectorItem, EuclideanDistance, DistanceCalculator}; // Added DistanceCalculator
use rand::Rng;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

// Constants from the HNSW paper
const M: usize = 16;  // Number of connections per layer
const EF_CONSTRUCTION: usize = 128;  // Size of dynamic candidate list during construction
const EF_SEARCH: usize = 64;  // Size of dynamic candidate list during search

fn generate_random_vector(dim: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn calculate_ground_truth(query: &VectorItem, vectors: &[VectorItem], k: usize) -> Vec<(usize, f64)> {
    let distance_calc = EuclideanDistance;
    let mut distances: Vec<(usize, f64)> = vectors
        .iter()
        .map(|v| (v.id, distance_calc.calculate(query, v)))  // Removed unwrap as it returns f64 directly
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.into_iter().take(k).collect()
}

fn main() {
    let n_vectors = 100_000;  // Number of vectors to index
    let dim = 128;           // Dimension of vectors
    let n_queries = 100;     // Number of queries to test
    let k = 10;             // Number of nearest neighbors to retrieve

    println!("=== HNSW Index Performance Test ===");
    println!("Parameters:");
    println!("  Vectors: {}", n_vectors);
    println!("  Dimensions: {}", dim);
    println!("  Queries: {}", n_queries);
    println!("  k-NN: {}", k);
    println!("  M: {}", M);
    println!("  EF Construction: {}", EF_CONSTRUCTION);
    println!("  EF Search: {}", EF_SEARCH);

    println!("\nInitializing HNSW index...");
    let hnsw = HnswIndex::new(Box::new(EuclideanDistance));

    println!("Generating {} random vectors...", n_vectors);
    let pb = ProgressBar::new(n_vectors as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("=>-"));

    let mut vectors = Vec::with_capacity(n_vectors);
    for i in 0..n_vectors {
        vectors.push(VectorItem {
            id: i,
            vector: generate_random_vector(dim),
        });
        pb.inc(1);
    }
    pb.finish_with_message("Vector generation complete");

    println!("\nBuilding index...");
    let build_start = Instant::now();
    match hnsw.batch_add(vectors.clone()) {
        Ok(_) => {
            let build_time = build_start.elapsed();
            println!("Index built successfully:");
            println!("  Build time: {:?}", build_time);
            println!("  Vectors per second: {:.2}", n_vectors as f64 / build_time.as_secs_f64());
        }
        Err(e) => {
            println!("Error building index: {}", e);
            return;
        }
    }

    let stats = hnsw.get_stats();
    println!("\nIndex Statistics:");
    println!("  Total nodes: {}", stats.total_nodes);
    println!("  Total connections: {}", stats.total_connections);
    println!("  Avg connections per node: {:.2}", 
             stats.total_connections as f64 / stats.total_nodes as f64);
    println!("  Max level: {}", stats.max_level);
    println!("  Level distribution:");
    let mut levels: Vec<_> = stats.level_distribution.into_iter().collect();
    levels.sort_by_key(|&(k, _)| k);
    for (level, count) in levels {
        println!("    Level {}: {} nodes", level, count);
    }

    println!("\nPerforming {} test queries...", n_queries);
    let pb = ProgressBar::new(n_queries as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("=>-"));

    let mut total_query_time = std::time::Duration::new(0, 0);
    let mut recall_sum = 0.0;
    let mut avg_precision_sum = 0.0;

    for _ in 0..n_queries {
        let query = VectorItem {
            id: n_vectors + 1,
            vector: generate_random_vector(dim),
        };

        // Get ground truth
        let ground_truth = calculate_ground_truth(&query, &vectors, k);
        
        // Perform HNSW search
        let query_start = Instant::now();
        match hnsw.search(&query, k) {
            Ok(results) => {
                total_query_time += query_start.elapsed();

                // Calculate recall (fraction of true nearest neighbors found)
                let result_ids: Vec<_> = results.iter().map(|r| r.id).collect();
                let truth_ids: Vec<_> = ground_truth.iter().map(|&(id, _)| id).collect();
                let correct = result_ids.iter()
                    .filter(|&&id| truth_ids.contains(&id))
                    .count();
                let recall = correct as f64 / k as f64;
                recall_sum += recall;

                // Calculate average precision
                let mut running_precision = 0.0;
                let mut correct_count = 0;
                for (i, &id) in result_ids.iter().enumerate() {
                    if truth_ids.contains(&id) {
                        correct_count += 1;
                        running_precision += correct_count as f64 / (i + 1) as f64;
                    }
                }
                let avg_precision = if correct_count > 0 {
                    running_precision / correct_count as f64
                } else {
                    0.0
                };
                avg_precision_sum += avg_precision;
            }
            Err(e) => println!("Search error: {}", e),
        }
        pb.inc(1);
    }
    pb.finish();

    let avg_query_time = total_query_time / n_queries as u32;
    let avg_recall = recall_sum / n_queries as f64;
    let mean_avg_precision = avg_precision_sum / n_queries as f64;

    println!("\nPerformance Metrics:");
    println!("  Average query time: {:?}", avg_query_time);
    println!("  Queries per second: {:.2}", 
             1.0 / avg_query_time.as_secs_f64());
    println!("  Average recall@{}: {:.4}", k, avg_recall);
    println!("  Mean average precision: {:.4}", mean_avg_precision);

    let memory_per_vector = dim * std::mem::size_of::<f64>();
    let memory_per_node = memory_per_vector + 
                         stats.total_connections * std::mem::size_of::<usize>() / stats.total_nodes;
    let total_memory = memory_per_node * stats.total_nodes;
    
    println!("\nMemory Usage Estimation:");
    println!("  Per vector: {} bytes", memory_per_vector);
    println!("  Per node avg: {} bytes", memory_per_node);
    println!("  Total: {:.2} MB", total_memory as f64 / 1024.0 / 1024.0);
}