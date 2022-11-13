use nalgebra::{DMatrix, MatrixXx1};

pub fn load_training_set() -> Result<(DMatrix<f64>, MatrixXx1<bool>), Box<dyn std::error::Error>> {
    let features_set = DMatrix::from_vec(20, 1, (0..20).map(|x| x as f64).collect::<Vec<f64>>());
    let target_set = MatrixXx1::from_vec(
        [
            (0..10).map(|_| false).collect::<Vec<bool>>(),
            (0..10).map(|_| true).collect::<Vec<bool>>(),
        ]
        .concat(),
    );
    Ok((features_set, target_set))
}
