use hnsw_rust::{HnswIndex, vector::{EuclideanDistance, VectorItem}};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

fn generate_random_vector(dim: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    let dist = Uniform::from(-1.0..1.0);
    (0..dim).map(|_| dist.sample(&mut rng)).collect()
}

fn main() {
    let hnsw = HnswIndex::new(Box::new(EuclideanDistance));  // Only pass the distance calculator
    
    let dimension = 10;
    let num_points = 100;
    
    println!("Inserting {} {}-dimensional vectors...", num_points, dimension);
    
    // Insert points one by one
    for i in 0..num_points {
        let vector = generate_random_vector(dimension);
        let item = VectorItem {
            id: i,
            vector,
        };
        
        if let Err(e) = hnsw.add(item) {
            eprintln!("Error adding item {}: {}", i, e);
            break;
        }
        
        if (i + 1) % 10 == 0 {
            println!("Inserted {} points", i + 1);
        }
    }
    
    println!("\nTesting search...");
    let query = VectorItem {
        id: 999,  // arbitrary id for query
        vector: generate_random_vector(dimension),
    };
    
    match hnsw.search(&query, 5) {
        Ok(results) => {
            println!("\nTop 5 nearest neighbors:");
            for (i, result) in results.iter().enumerate() {
                println!("#{}: Vector ID {}", i + 1, result.id);
            }
        }
        Err(e) => eprintln!("Search error: {}", e),
    }
}