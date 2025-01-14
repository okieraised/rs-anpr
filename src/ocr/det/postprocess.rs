pub struct Postprocess {
    threshold: f32,
    min_size: usize,
}

impl Postprocess {
    pub fn new(threshold: f32, min_size: usize) -> Self {
        Postprocess {
            threshold,
            min_size,
        }
    }


}