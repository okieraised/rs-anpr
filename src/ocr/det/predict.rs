use std::cmp::max;
use anyhow::Error;
use ndarray::{Array1, Array3};
use opencv::core::{copy_make_border, Mat, MatTraitConst, Scalar, Size, Vec3b, Vector, BORDER_CONSTANT};
use opencv::imgcodecs::imwrite;
use opencv::imgproc::{resize, threshold, INTER_LINEAR};
use tract_onnx::prelude::{tract_ndarray, tvec, Datum, DatumExt, Framework, Graph, InferenceFact, InferenceModelExt, RunnableModel, Tensor, TypedFact, TypedOp};
use crate::ocr::det::postprocess::Postprocess;
use crate::ocr::det::preprocess::Preprocess;

pub struct TextDetector {
    model_path: String,
    bbox_threshold: f32,
    unclip_ratio: f32,
    preprocess: Preprocess,
    postprocess: Postprocess,
    // model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>
}


impl TextDetector {

    pub fn new(model_path: &str, bbox_threshold: Option<f32>, unclip_ratio: Option<f32>) -> Result<Self, Error> {
        let bbox_threshold = bbox_threshold.unwrap_or(0.6);
        let unclip_ratio= unclip_ratio.unwrap_or(1.6);
        let preprocess = Preprocess::new();
        let postprocess = Postprocess::new(0.3, 3);


        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_typed()?
            .with_input_fact(0, f32::fact([1, 3, 224, 224]).into())?
            .into_optimized()?
            .into_runnable()?;

        println!("model: {:?}", model);


        Ok(TextDetector {
            model_path: model_path.to_string(),
            bbox_threshold,
            unclip_ratio,
            preprocess,
            postprocess,
            // model,
        })
    }

    pub fn call(&self, img: &Mat, bbox_threshold: Option<f32>, unclip_ratio: Option<f32>) -> Result<(), Error> {
        let bbox_threshold = bbox_threshold.unwrap_or(self.bbox_threshold);
        let unclip_ratio= unclip_ratio.unwrap_or(self.unclip_ratio);

        let (resized, shape) = self.preprocess.resize(img, 960)?;


        // let (preprocessed, shape) = self.

        Ok(())
    }

}

#[cfg(test)]
mod tests {
    use crate::utils::image::convert_image_to_mat;
    use super::*;

    #[test]
    fn test_resize() {
        let im_bytes: &[u8] = include_bytes!("../../../test_data/car.jpg");
        let image = convert_image_to_mat(im_bytes).unwrap();
        let detector = TextDetector::new("plate_det_infer.onnx", None, None).unwrap();
        detector.preprocess.resize(&image, 960).unwrap();
    }

    #[test]
    fn test_normalize() {
        let im_bytes: &[u8] = include_bytes!("../../../test_data/car.jpg");
        let image = convert_image_to_mat(im_bytes).unwrap();
        let detector = TextDetector::new("plate_det_infer.onnx", None, None).unwrap();

        let (img, shape) = detector.preprocess.resize(&image, 960).unwrap();
        detector.preprocess.normalize(&img, vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225], 1f32/255f32);
    }

    #[test]
    fn test_hwc_to_chw() {
        let im_bytes: &[u8] = include_bytes!("../../../test_data/car.jpg");
        let image = convert_image_to_mat(im_bytes).unwrap();
        let detector = TextDetector::new("plate_det_infer.onnx", None, None).unwrap();

        let (img, shape) = detector.preprocess.resize(&image, 960).unwrap();
        let tensors = detector.preprocess.normalize(&img, vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225], 1f32/255f32).unwrap();
        let tensor = detector.preprocess.hwc_to_chw(tensors);

        println!("tensor: {:?}", tensor);
    }

}