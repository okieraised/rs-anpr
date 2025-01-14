use std::fmt;
use opencv::core::Mat;


pub struct Pipeline {
    bbox_threshold: f32,
    unclip_ratio: f32,
    text_det_model: String,
    text_rec_model: String,
}

impl Pipeline {
    pub fn new(bbox_threshold: Option<f32>, unclip_ratio: Option<f32>, text_det_model: Option<String>, text_rec: Option<String>, text_rec_dict: Option<String>) -> Self {
        Pipeline {
            bbox_threshold: 0.0,
            unclip_ratio: 0.0,
            text_det_model: "".to_string(),
            text_rec_model: "".to_string(),
        }
    }

    pub fn detect_and_ocr(&self, img: &Mat, drop_score: Option<f32>, unclip_ration: Option<f32>, bbox_threshold: Option<f32>) {

    }
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


