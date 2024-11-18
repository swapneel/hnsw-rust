// src/bin/vector_generator.rs
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, thread_rng};

fn generate_random_vector(dim: usize, rng: &mut impl Rng) -> Vec<f64> {
    (0..dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect()
}

fn create_test_vectors(
    output_dir: &Path,
    n_files: usize,
    vectors_per_file: usize,
    dimensions: usize,
) -> std::io::Result<()> {
    fs::create_dir_all(output_dir)?;
    let mut rng = thread_rng();

    let pb = ProgressBar::new((n_files * vectors_per_file) as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap()
        .progress_chars("=>-"));

    // Create files with different "types" of vectors
    for file_idx in 0..n_files {
        let file_path = output_dir.join(format!("vectors_{:03}.txt", file_idx));
        let mut file = File::create(file_path)?;

        // Create a "center" for this file's vectors
        let center: Vec<f64> = generate_random_vector(dimensions, &mut rng);
        
        // Generate vectors clustered around this center
        for _ in 0..vectors_per_file {
            let mut vector = Vec::with_capacity(dimensions);
            
            // Generate vector components with some noise around the center
            for i in 0..dimensions {
                let noise = rng.gen_range(-0.2..0.2);
                vector.push(center[i] + noise);
            }

            // Write vector to file with high precision
            for (i, component) in vector.iter().enumerate() {
                if i > 0 {
                    write!(file, " ")?;
                }
                write!(file, "{:.8}", component)?;
            }
            writeln!(file)?;
            
            pb.inc(1);
        }
        
        pb.set_message(format!("Generated file {}/{}", file_idx + 1, n_files));
    }

    pb.finish_with_message("Vector generation complete");
    Ok(())
}

fn main() -> std::io::Result<()> {
    let output_dir = Path::new("test_vectors");
    let n_files = 5;           // Number of files to generate
    let vectors_per_file = 100; // Vectors in each file
    let dimensions = 128;       // Dimensions per vector

    println!("Vector Generation Tool");
    println!("--------------------");
    println!("Output directory: {}", output_dir.display());
    println!("Files to generate: {}", n_files);
    println!("Vectors per file: {}", vectors_per_file);
    println!("Vector dimensions: {}", dimensions);
    println!("Total vectors: {}", n_files * vectors_per_file);
    println!();

    create_test_vectors(output_dir, n_files, vectors_per_file, dimensions)?;

    println!("\nTest vectors have been generated!");
    Ok(())
}