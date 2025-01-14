pub struct PostProcess {
    threshold: f32,
    min_size: usize,
}

impl PostProcess {
    fn new(threshold: f32, min_size: usize) -> Self {
        PostProcess {
            threshold,
            min_size,
        }
    }


}