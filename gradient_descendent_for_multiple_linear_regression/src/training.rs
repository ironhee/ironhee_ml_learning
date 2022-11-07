use crate::model::{EstimationModel, Gradient};
use nalgebra::{DMatrix, MatrixXx1, RowDVector};
use num_traits::pow;
use std::ops::{Div, DivAssign};

pub fn estimate(features: &RowDVector<f64>, estimation_model: &EstimationModel) -> f64 {
    estimation_model.parameters.dot(&features) + estimation_model.b
}

pub fn get_row_error(
    features: &RowDVector<f64>,
    target: &f64,
    estimation_model: &EstimationModel,
) -> f64 {
    estimate(features, estimation_model) - target
}

fn get_cost_of_rows_1(
    features_set: &DMatrix<f64>,
    target_set: &MatrixXx1<f64>,
    estimation_model: &EstimationModel,
) -> f64 {
    let row_count = features_set.len() / estimation_model.parameters.len();
    let total_cost_of_rows = features_set
        .row_iter()
        .enumerate()
        .map(|(index, features)| {
            pow(
                get_row_error(
                    &RowDVector::from(features),
                    &target_set[index],
                    estimation_model,
                ),
                2,
            )
        })
        .sum::<f64>();
    total_cost_of_rows / (2.0 * row_count as f64)
}

pub fn get_gradient(
    features_set: &DMatrix<f64>,
    target_set: &MatrixXx1<f64>,
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
    target_set: &MatrixXx1<f64>,
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

pub fn zscore_normalize_features(
    features: &RowDVector<f64>,
    row_mean: &RowDVector<f64>,
    row_std_deviation: &RowDVector<f64>,
) -> RowDVector<f64> {
    let mut normalized_features = features.clone();
    for (i, mut column) in normalized_features.column_iter_mut().enumerate() {
        column.add_scalar_mut(row_mean[i] * -1.0);
        column.div_assign(row_std_deviation[i]);
    }
    normalized_features
}

pub fn zscore_normalize_features_set(
    features_set: &DMatrix<f64>,
    row_mean: &RowDVector<f64>,
    row_std_deviation: &RowDVector<f64>,
) -> DMatrix<f64> {
    let mut normalized_feature_set = features_set.clone();
    for (i, mut column) in normalized_feature_set.column_iter_mut().enumerate() {
        column.add_scalar_mut(row_mean[i] * -1.0);
        column.div_assign(row_std_deviation[i]);
    }
    normalized_feature_set
}

pub fn get_row_mean(features_set: &DMatrix<f64>) -> RowDVector<f64> {
    features_set.row_mean()
}

pub fn get_row_std_deviation(
    features_set: &DMatrix<f64>,
    row_mean: &RowDVector<f64>,
) -> RowDVector<f64> {
    RowDVector::from_vec(
        features_set
            .column_iter()
            .enumerate()
            .map(|(index, column)| {
                column
                    .iter()
                    .map(|feature| pow(row_mean[index] - feature, 2))
                    .sum::<f64>()
                    .div(column.len() as f64)
                    .sqrt()
            })
            .collect::<Vec<_>>(),
    )
}
