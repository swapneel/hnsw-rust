use crate::vector::VectorItem;

pub struct Node {
    pub id: usize,
    pub connections: Vec<Vec<usize>>,
    pub item: VectorItem,
    pub layer: usize,
}
