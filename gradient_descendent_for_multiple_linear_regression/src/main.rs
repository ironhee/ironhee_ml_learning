mod data_loader;
mod model;
mod plot;
mod training;
use crate::data_loader::load_training_set;
use crate::model::{EstimationModel, SeoulRealEstateTransactionPrice};
use crate::plot::draw_costs_plot;
use crate::training::{
    estimate, get_row_mean, get_row_std_deviation, gradient_descent, zscore_normalize_features,
    zscore_normalize_features_set,
};
use chrono::NaiveDate;
use nalgebra::RowDVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (features_set, target_set) = load_training_set("seoul_real_estate_transaction_price.csv")?;
    let row_mean = get_row_mean(&features_set);
    let row_std_deviation = get_row_std_deviation(&features_set, &row_mean);
    let features_set = zscore_normalize_features_set(&features_set, &row_mean, &row_std_deviation);

    let learning_rate = 1.0e-1;
    let iteration_count = 100;
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

    draw_costs_plot("plot/training.png", learning_rate, &costs)?;

    // Test
    for test_data in vec![
        SeoulRealEstateTransactionPrice {
            계약일: NaiveDate::from_ymd(2015, 6, 2),
            물건금액: None,
            건물면적: 10.0,
            토지면적: 20.0,
        },
        SeoulRealEstateTransactionPrice {
            계약일: NaiveDate::from_ymd(2015, 6, 2),
            물건금액: None,
            건물면적: 39.54,
            토지면적: 53.33,
        },
        SeoulRealEstateTransactionPrice {
            계약일: NaiveDate::from_ymd(2015, 6, 2),
            물건금액: None,
            건물면적: 80.0,
            토지면적: 100.0,
        },
        SeoulRealEstateTransactionPrice {
            계약일: NaiveDate::from_ymd(2015, 6, 2),
            물건금액: None,
            건물면적: 160.0,
            토지면적: 200.0,
        },
    ]
    .iter()
    {
        println!(
            "{:?} 의 가격 예상가: {:?}만원",
            test_data,
            estimate(
                &zscore_normalize_features(
                    &RowDVector::from_vec(test_data.into_features()),
                    &row_mean,
                    &row_std_deviation
                ),
                &result
            ) as usize
        );
    }

    Ok(())
}
