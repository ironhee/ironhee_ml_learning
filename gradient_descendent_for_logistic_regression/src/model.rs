use nalgebra::RowDVector;

#[derive(Debug, Clone)]
pub struct EstimationModel {
    pub parameters: RowDVector<f64>,
    pub b: f64,
}

#[derive(Debug)]
pub struct Gradient {
    pub parameters: RowDVector<f64>,
    pub b: f64,
}
