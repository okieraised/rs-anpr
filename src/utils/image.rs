use anyhow::Error;
use opencv::core::Mat;
use opencv::imgcodecs::{imdecode, IMREAD_COLOR};

pub fn convert_image_to_mat(im_bytes: &[u8]) -> Result<Mat, Error> {
    // Convert bytes to Mat
    let img_as_mat = Mat::from_slice(im_bytes)?;

    // Decode the image
    let img = imdecode(&img_as_mat, IMREAD_COLOR)?;

    Ok(img)
}