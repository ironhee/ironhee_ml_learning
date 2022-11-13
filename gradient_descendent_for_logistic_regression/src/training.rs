use crate::model::{EstimationModel, Gradient};
use nalgebra::{DMatrix, MatrixXx1, RowDVector};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (x * -1.0).exp())
}

pub fn estimate(features: &RowDVector<f64>, estimation_model: &EstimationModel) -> f64 {
    sigmoid(estimation_model.parameters.dot(&features) + estimation_model.b)
}

pub fn get_row_loss(
    features: &RowDVector<f64>,
    target: &bool,
    estimation_model: &EstimationModel,
) -> f64 {
    let estimated = estimate(features, estimation_model);
    let target_f64 = match target {
        true => 1.0,
        false => 0.0,
    };
    -(target_f64 * estimated.ln()) + -((1.0 - target_f64) * (1.0 - estimated).ln())
}

pub fn get_row_error(
    features: &RowDVector<f64>,
    target: &bool,
    estimation_model: &EstimationModel,
) -> f64 {
    let target_f64 = match target {
        true => 1.0,
        false => 0.0,
    };
    estimate(features, estimation_model) - target_f64
}

fn get_cost_of_rows_1(
    features_set: &DMatrix<f64>,
    target_set: &MatrixXx1<bool>,
    estimation_model: &EstimationModel,
) -> f64 {
    let row_count = features_set.len() / estimation_model.parameters.len();
    let total_count = features_set
        .row_iter()
        .enumerate()
        .map(|(index, features)| {
            get_row_loss(
                &RowDVector::from(features),
                &target_set[index],
                estimation_model,
            )
        })
        .sum::<f64>();
    total_count / row_count as f64
}

pub fn get_gradient(
    features_set: &DMatrix<f64>,
    target_set: &MatrixXx1<bool>,
    estimation_model: &EstimationModel,
) -> Gradient {
    let row_count = features_set.len() / estimation_model.parameters.len();
    let mut parameter_gradients = RowDVector::zeros(estimation_model.parameters.len());
    let mut b_gratient = 0.0;

    for (index, features) in features_set.row_iter().enumerate() {
        let cost_of_row = get_row_error(
            &RowDVector::from(features),
            &target_set[index],
            estimation_model,
        );
        parameter_gradients += cost_of_row * features;
        b_gratient += cost_of_row;
    }
    parameter_gradients /= row_count as f64;
    b_gratient /= row_count as f64;

    Gradient {
        parameters: parameter_gradients,
        b: b_gratient,
    }
}

pub fn gradient_descent(
    features_set: &DMatrix<f64>,
    target_set: &MatrixXx1<bool>,
    initial_estimation_model: &EstimationModel,
    learning_rate: f64,
    iteration_count: usize,
) -> (EstimationModel, Vec<f64>) {
    let mut costs: Vec<f64> = Vec::new();
    let mut estimation_model = initial_estimation_model.clone();

    for _ in 0..iteration_count.clone() {
        let gradient = get_gradient(features_set, target_set, &estimation_model);
        let cost = get_cost_of_rows_1(features_set, target_set, &estimation_model);

        costs.push(cost);

        estimation_model = EstimationModel {
            parameters: estimation_model.parameters - learning_rate * gradient.parameters,
            b: estimation_model.b - learning_rate * gradient.b,
        }
    }

    (estimation_model, costs)
}
