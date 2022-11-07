use crate::model::SeoulRealEstateTransactionPrice;
use nalgebra::{DMatrix, MatrixXx1};
use std::fs::File;

pub fn load_training_set(
    file_path: &str,
) -> Result<(DMatrix<f64>, MatrixXx1<f64>), Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_reader(File::open(file_path)?);
    let mut features_vec: Vec<Vec<f64>> = Vec::new();
    let mut target_vec: Vec<f64> = Vec::new();

    for result in rdr.deserialize::<SeoulRealEstateTransactionPrice>() {
        match result {
            Ok(record) => {
                features_vec.push(record.into_features());
                target_vec.push(record.into_target());
            }
            Err(_) => {
                continue;
            }
        }
    }
    let features_set = DMatrix::from_vec(
        features_vec.len(),
        features_vec[0].len(),
        features_vec.concat(),
    );
    let target_set = MatrixXx1::from_vec(target_vec);

    Ok((features_set, target_set))
}
