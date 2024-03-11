#[derive(Clone, Debug)]
pub struct VectorItem {
    pub id: usize,
    pub vector: Vec<f64>,
}

pub trait DistanceCalculator {
    fn calculate(&self, item1: &VectorItem, item2: &VectorItem) -> f64;
}

pub struct EuclideanDistance;

impl DistanceCalculator for EuclideanDistance {
    fn calculate(&self, item1: &VectorItem, item2: &VectorItem) -> f64 {
        item1.vector.iter().zip(item2.vector.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}
