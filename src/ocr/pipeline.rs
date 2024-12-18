use std::fmt;
use opencv::core::Mat;


pub struct Pipeline {
    bbox_threshold: f32,
    unclip_ratio: f32,
    text_det_model: String,
    text_rec_model: String,
}



pub type BoundingBox = (f32, f32, f32, f32);

pub struct BoxedResult {
    pub bbox: BoundingBox,
    pub img: Mat,
    pub text: String,
    pub score: f32,
}

impl BoxedResult {
    pub fn new(bbox: BoundingBox, img: Mat, text: &str, score: f32) -> Self {
        Self {
            bbox,
            img,
            text: text.to_string(),
            score,
        }
    }
}

impl fmt::Debug for BoxedResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}[{}, {:.2}]", std::any::type_name::<Self>(), self.text, self.score)
    }
}


