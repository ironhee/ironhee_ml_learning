mod data_loader;
mod model;
mod plot;
mod training;

use crate::data_loader::load_training_set;
use crate::model::EstimationModel;
use crate::plot::draw_costs_plot;
use crate::training::{estimate, gradient_descent};
use nalgebra::RowDVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (features_set, target_set) = load_training_set()?;

    let learning_rate = 1.0e-1;
    let iteration_count = 1000;
    let initial = EstimationModel {
        parameters: RowDVector::zeros(features_set.row(0).len()),
        b: 0.0,
    };

    let (result, costs) = gradient_descent(
        &features_set,
        &target_set,
        &initial,
        learning_rate,
        iteration_count,
    );

    // Test
    for test_data in 0..20 {
        println!(
            "{:?} 예상 결과: {:?}",
            test_data,
            estimate(&RowDVector::from_vec(vec![test_data as f64]), &result)
        );
    }

    draw_costs_plot("plot/training.png", learning_rate, &costs)?;

    Ok(())
}
