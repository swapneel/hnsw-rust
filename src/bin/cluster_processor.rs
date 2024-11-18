use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write, BufRead};
use std::path::Path;
use std::collections::HashMap;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use hnsw_rust::{HnswIndex, VectorItem, EuclideanDistance};

#[derive(Debug)]
struct Args {
    input: String,
    output: String,
    clusters: usize,
}

impl Args {
    fn from_env() -> Self {
        let args: Vec<String> = std::env::args().collect();
        
        Args {
            input: args.get(1)
                .cloned()
                .unwrap_or_else(|| "test_vectors".to_string()),
            output: args.get(2)
                .cloned()
                .unwrap_or_else(|| "clustered_vectors".to_string()),
            clusters: args.get(3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct VectorFile {
    vectors: Vec<Vec<f64>>,
    filenames: Vec<String>,
}

struct ClusterProcessor {
    index: HnswIndex,
    vector_map: HashMap<usize, (Vec<f64>, String)>, 
    cluster_map: HashMap<usize, Vec<usize>>,        
    processed_count: usize,
    k_clusters: usize,
}

impl ClusterProcessor {
    fn new(k_clusters: usize) -> Self {
        ClusterProcessor {
            index: HnswIndex::new(Box::new(EuclideanDistance)),
            vector_map: HashMap::new(),
            cluster_map: HashMap::new(),
            processed_count: 0,
            k_clusters,
        }
    }

    fn process_directory(&mut self, dir_path: &Path) -> std::io::Result<()> {
        let total_files = fs::read_dir(dir_path)?.count();
        let pb = ProgressBar::new(total_files as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("=>-"));

        for entry in fs::read_dir(dir_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                self.process_file(&path)?;
                pb.inc(1);
                pb.set_message(format!("File: {}", path.display()));
            }
        }
        pb.finish_with_message("Directory processing complete");

        Ok(())
    }

    fn process_file(&mut self, file_path: &Path) -> std::io::Result<()> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let filename = file_path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();

        for line in reader.lines() {
            let line = line?;
            let vector: Vec<f64> = line
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if !vector.is_empty() {
                let id = self.processed_count;
                self.vector_map.insert(id, (vector.clone(), filename.clone()));
                
                self.index.add(VectorItem {
                    id,
                    vector: vector.clone(),
                }).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

                self.processed_count += 1;
            }
        }

        Ok(())
    }

    fn cluster_vectors(&mut self) -> std::io::Result<()> {
        println!("\nClustering {} vectors into {} clusters...", self.processed_count, self.k_clusters);
        let pb = ProgressBar::new(self.processed_count as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("=>-"));

        // Initialize cluster centers
        for i in 0..self.k_clusters {
            if let Some((_vector, _)) = self.vector_map.get(&i) {
                self.cluster_map.insert(i, vec![i]);
                pb.inc(1);
            }
        }

        // Assign vectors to clusters
        for id in self.k_clusters..self.processed_count {
            if let Some((vector, _)) = self.vector_map.get(&id) {
                let query = VectorItem {
                    id,
                    vector: vector.clone(),
                };

                if let Ok(nearest) = self.index.search(&query, 1) {
                    if let Some(nearest) = nearest.first() {
                        let cluster_id = nearest.id % self.k_clusters;
                        self.cluster_map.entry(cluster_id)
                            .or_insert_with(Vec::new)
                            .push(id);
                    }
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("Clustering complete");

        Ok(())
    }

    fn write_clusters(&self, output_dir: &Path) -> std::io::Result<()> {
        fs::create_dir_all(output_dir)?;
        println!("\nWriting clusters to {}", output_dir.display());
        
        let pb = ProgressBar::new(self.k_clusters as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("=>-"));

        for (cluster_id, vector_ids) in &self.cluster_map {
            let output_path = output_dir.join(format!("cluster_{:05}.txt", cluster_id));
            let mut writer = BufWriter::new(File::create(&output_path)?);

            writeln!(writer, "# Cluster {} - {} vectors", cluster_id, vector_ids.len())?;
            writeln!(writer, "# Format: vector_components | original_filename")?;

            for &id in vector_ids {
                if let Some((vector, filename)) = self.vector_map.get(&id) {
                    // Write vector components
                    for component in vector {
                        write!(writer, "{:.6} ", component)?;
                    }
                    writeln!(writer, "| {}", filename)?;
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("Cluster files written");

        let stats_path = output_dir.join("cluster_stats.txt");
        let mut stats_writer = BufWriter::new(File::create(stats_path)?);
        writeln!(stats_writer, "Clustering Statistics")?;
        writeln!(stats_writer, "--------------------")?;
        writeln!(stats_writer, "Total vectors: {}", self.processed_count)?;
        writeln!(stats_writer, "Number of clusters: {}", self.k_clusters)?;
        writeln!(stats_writer, "\nCluster sizes:")?;
        
        let mut sizes: Vec<_> = self.cluster_map
            .iter()
            .map(|(id, vectors)| (*id, vectors.len()))
            .collect();
        sizes.sort_by_key(|&(_, size)| std::cmp::Reverse(size));

        for (cluster_id, size) in sizes {
            writeln!(stats_writer, "Cluster {:5}: {:6} vectors", cluster_id, size)?;
        }

        Ok(())
    }
}

fn main() {
    let args = Args::from_env();

    println!("Vector Clustering Tool");
    println!("--------------------");
    println!("Input directory:  {}", args.input);
    println!("Output directory: {}", args.output);
    println!("Number of clusters: {}", args.clusters);
    println!("\nUse: cargo run --bin cluster-processor <input_dir> <output_dir> <num_clusters>");

    let mut processor = ClusterProcessor::new(args.clusters);

    if let Err(e) = processor.process_directory(Path::new(&args.input)) {
        eprintln!("Error processing directory: {}", e);
        return;
    }

    if let Err(e) = processor.cluster_vectors() {
        eprintln!("Error during clustering: {}", e);
        return;
    }

    if let Err(e) = processor.write_clusters(Path::new(&args.output)) {
        eprintln!("Error writing clusters: {}", e);
        return;
    }

    println!("\nProcessing complete! Check {} for results", args.output);
}