use std::cmp::max;
use anyhow::Error;
use ndarray::{Array1, Array3};
use opencv::core::{copy_make_border, Mat, MatTraitConst, Scalar, Size, Vec3b, Vector, BORDER_CONSTANT};
use opencv::imgcodecs::imwrite;
use opencv::imgproc::{resize, INTER_LINEAR};

pub struct Preprocess {
}

impl Preprocess {
    pub fn new() -> Self {
        Preprocess {}
    }
    pub fn resize(&self, img: &Mat, limit_side_len: i32) -> Result<(Mat, Array1<f32>), Error> {
        let src_h = img.rows();
        let src_w = img.cols();

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

    pub fn normalize(&self, img: &Mat, mean: Vec<f32>, std: Vec<f32>, scale: f32) -> Result<Array3<f32>, Error>{
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

    pub fn hwc_to_chw(&self, img: Array3<f32>) -> Array3<f32> {
        img.permuted_axes([2, 0, 1])
    }
}