use std::cmp::max;
use anyhow::Error;
use ndarray::{Array1, Array3};
use opencv::core::{copy_make_border, Mat, MatTraitConst, Scalar, Size, Vec3b, Vector, BORDER_CONSTANT};
use opencv::imgcodecs::imwrite;
use opencv::imgproc::{resize, INTER_LINEAR};
use tract_onnx::prelude::{tract_ndarray, Tensor};

pub struct TextDetector {
    model_path: String,
    bbox_threshold: f32,
    unclip_ratio: f32,
}


impl TextDetector {

    pub fn new(model_path: &str, bbox_threshold: Option<f32>, unclip_ratio: Option<f32>) -> Self {
        let bbox_threshold = bbox_threshold.unwrap_or(0.6);
        let unclip_ratio= unclip_ratio.unwrap_or(1.6);

        TextDetector {
            model_path: model_path.to_string(),
            bbox_threshold,
            unclip_ratio,
        }
    }

    pub fn call(&self, img: &Mat, bbox_threshold: Option<f32>, unclip_ratio: Option<f32>) -> Result<(), Error> {
        let bbox_threshold = bbox_threshold.unwrap_or(self.bbox_threshold);
        let unclip_ratio= unclip_ratio.unwrap_or(self.unclip_ratio);

        // let (preprocessed, shape) = self.

        Ok(())
    }

    fn resize(&self, img: &Mat, limit_side_len: i32) -> Result<(Mat, Array1<f32>), Error> {
        let src_h = img.rows();
        let src_w = img.cols();
        println!("here {}, {}", src_h, src_w);

        let mut ratio = 1.0;
        let mut resized_img = img.clone();
        if max(src_h, src_w) > limit_side_len {
            ratio = if src_h > src_w {
                limit_side_len as f32 / src_h as f32
            } else {
                limit_side_len as f32 / src_w as f32
            };

            let new_width = (ratio * src_w as f32).round() as i32;
            let new_height = (ratio * src_h as f32).round() as i32;
            let size = Size::new(new_width, new_height);
            resized_img = Mat::default();
            resize(&img, &mut resized_img, size, 0.0, 0.0, INTER_LINEAR)?;
        };

        let padding_top = 0;
        let padding_bottom = limit_side_len - resized_img.rows();
        let padding_left = 0;
        let padding_right = limit_side_len - resized_img.cols();
        let mut padded_img = Mat::default();
        copy_make_border(
            &resized_img,
            &mut padded_img,
            padding_top,
            padding_bottom,
            padding_left,
            padding_right,
            BORDER_CONSTANT,
            Scalar::new(0.0, 0.0, 0.0, 0.0),
        )?;
        drop(resized_img);

        imwrite("./test.jpeg", &padded_img, &Vector::new())?;

        let shape = Array1::from(vec![src_h as f32, src_w as f32, ratio]);

        Ok((padded_img, shape))
    }

    fn normalize(&self, img: &Mat, mean: Vec<f32>, std: Vec<f32>, scale: f32) -> Result<Array3<f32>, Error>{
        let src_h = img.rows();
        let src_w = img.cols();
        let mut tensors = Array3::<f32>::zeros((src_w as usize,src_h as usize, 3usize));

        for c in 0..3 {
            for y in 0..src_w as usize {
                for x in 0..src_h as usize {
                    let pixel_value = img.at_2d::<Vec3b>(y as i32, x as i32).unwrap()[c];
                    tensors[[y, x, c]] = (pixel_value as f32 * scale - mean[c]) / std[c];
                }
            }
        }
        Ok(tensors)
    }

    fn hwc_to_chw(&self, img: Array3<f32>) -> Array3<f32> {
        img.permuted_axes([2, 0, 1])
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
        let detector = TextDetector::new("", None, None);
        detector.resize(&image, 960).unwrap();
    }

    #[test]
    fn test_normalize() {
        let im_bytes: &[u8] = include_bytes!("../../../test_data/car.jpg");
        let image = convert_image_to_mat(im_bytes).unwrap();
        let detector = TextDetector::new("../../../weights/ppocrv4/plate_det_infer.onnx", None, None);

        let (img, shape) = detector.resize(&image, 960).unwrap();
        detector.normalize(&img, vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225], 1f32/255f32);
    }

    #[test]
    fn test_hwc_to_chw() {
        let im_bytes: &[u8] = include_bytes!("../../../test_data/car.jpg");
        let image = convert_image_to_mat(im_bytes).unwrap();
        let detector = TextDetector::new("../../../weights/ppocrv4/plate_det_infer.onnx", None, None);

        let (img, shape) = detector.resize(&image, 960).unwrap();
        let tensors = detector.normalize(&img, vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225], 1f32/255f32).unwrap();
        let tensor = detector.hwc_to_chw(tensors);

        println!("tensor: {:?}", tensor);
    }

}