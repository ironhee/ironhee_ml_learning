extern crate nalgebra as na;
use chrono::prelude::*;
use chrono::NaiveDate;
use na::DVector;
use num_traits::pow;
use plotters::prelude::*;
use serde::Deserialize;
use std::fs::File;

#[derive(Debug, Deserialize, Clone, Copy)]
struct SeoulRealEstateTransactionPrice {
    #[serde(with = "datetime_ymd_format")]
    계약일: NaiveDate,
    #[serde(rename(deserialize = "물건금액(만원)"))]
    물건금액: f64,
    #[serde(rename(deserialize = "건물면적(㎡)"))]
    건물면적: f64,
    #[serde(rename(deserialize = "토지면적(㎡)"))]
    토지면적: f64,
}

mod datetime_ymd_format {
    use chrono::NaiveDate;
    use serde::{self, Deserialize, Deserializer};

    const FORMAT: &'static str = "%Y%m%d";

    pub fn deserialize<'de, D>(deserializer: D) -> Result<NaiveDate, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        NaiveDate::parse_from_str(&s, FORMAT).map_err(serde::de::Error::custom)
    }
}

impl SeoulRealEstateTransactionPrice {
    fn to_training_row(self) -> TrainingRow {
        TrainingRow {
            features: DVector::from_vec(vec![
                self.건물면적,
                f64::from(self.계약일.num_days_from_ce()) / 100000.0,
                self.토지면적 / 100.0,
            ]),
            target: self.물건금액,
        }
    }
}

#[derive(Debug)]
struct TrainingRow {
    features: DVector<f64>,
    target: f64,
}

#[derive(Debug, Clone)]
struct EstimationModel {
    parameters: DVector<f64>,
    b: f64,
}

#[derive(Debug)]
struct Gradient {
    parameters: DVector<f64>,
    b: f64,
}

fn estimate(training_row: &TrainingRow, estimation_model: &EstimationModel) -> f64 {
    estimation_model.parameters.dot(&training_row.features) + estimation_model.b
}

fn get_cost_of_row(training_row: &TrainingRow, estimation_model: &EstimationModel) -> f64 {
    estimate(training_row, estimation_model) - training_row.target
}

fn get_cost_of_rows_1(training_rows: &[TrainingRow], estimation_model: &EstimationModel) -> f64 {
    let row_count = training_rows.len();
    let total_cost_of_rows = training_rows
        .iter()
        .map(|row| pow(get_cost_of_row(row, estimation_model), 2))
        .sum::<f64>();
    total_cost_of_rows / (2.0 * row_count as f64)
}

fn get_gradient(training_rows: &[TrainingRow], estimation_model: &EstimationModel) -> Gradient {
    let row_count = training_rows.len();
    let mut parameter_gradients = DVector::zeros(estimation_model.parameters.len());
    let mut b_gratient = 0.0;

    for training_row in training_rows {
        let cost_of_row = get_cost_of_row(training_row, estimation_model);
        for (index, feature) in training_row.features.iter().enumerate() {
            parameter_gradients[index] += cost_of_row * feature;
        }
        b_gratient += cost_of_row;
    }
    parameter_gradients /= row_count as f64;
    b_gratient /= row_count as f64;

    Gradient {
        parameters: parameter_gradients,
        b: b_gratient,
    }
}

fn gradient_descent(
    training_rows: &[TrainingRow],
    initial_estimation_model: &EstimationModel,
    learning_rate: f64,
    iteration_count: usize,
) -> (EstimationModel, Vec<f64>) {
    let mut costs: Vec<f64> = Vec::new();
    let mut estimation_model = initial_estimation_model.clone();

    for _ in 0..iteration_count.clone() {
        let gradient = get_gradient(training_rows, &estimation_model);
        let cost = get_cost_of_rows_1(training_rows, &estimation_model);
        costs.push(cost);

        estimation_model = EstimationModel {
            parameters: estimation_model.parameters - learning_rate * gradient.parameters,
            b: estimation_model.b - learning_rate * gradient.b,
        }
    }

    (estimation_model, costs)
}

fn draw_costs_plot(
    filename: &str,
    learning_rate: f64,
    costs: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    // Plot
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(
            format!("Learning Rate: {}", learning_rate),
            ("sans-serif", 20),
        )
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0..costs.len(), 0.0..(costs[0] * 1.2))?;
    chart
        .configure_mesh()
        .x_desc("iteration")
        .y_desc("cost of function")
        .draw()?;
    chart.draw_series(LineSeries::new(
        costs
            .iter()
            .enumerate()
            .map(|(index, cost)| (index, cost.clone())),
        &RED,
    ))?;
    chart.configure_series_labels().draw()?;
    root.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_reader(File::open("seoul_real_estate_transaction_price.csv")?);
    let mut training_rows = Vec::<TrainingRow>::new();

    for result in rdr.deserialize::<SeoulRealEstateTransactionPrice>() {
        match result {
            Ok(record) => training_rows.push(record.to_training_row()),
            Err(_) => {
                continue;
            }
        }
    }

    let learning_rate = 5.0e-7;
    let iteration_count = 1000;
    let initial = EstimationModel {
        parameters: DVector::zeros(training_rows[0].features.nrows()),
        b: 0.0,
    };

    let (result, costs) =
        gradient_descent(&training_rows, &initial, learning_rate, iteration_count);

    draw_costs_plot("plot/training.png", learning_rate, &costs)?;

    for data in vec![
        SeoulRealEstateTransactionPrice {
            계약일: NaiveDate::from_ymd(2015, 6, 2),
            물건금액: 0.0,
            건물면적: 10.0,
            토지면적: 20.0,
        },
        SeoulRealEstateTransactionPrice {
            계약일: NaiveDate::from_ymd(2015, 6, 2),
            물건금액: 0.0,
            건물면적: 39.54,
            토지면적: 53.33,
        },
        SeoulRealEstateTransactionPrice {
            계약일: NaiveDate::from_ymd(2015, 6, 2),
            물건금액: 0.0,
            건물면적: 39.54,
            토지면적: 53.33,
        },
        SeoulRealEstateTransactionPrice {
            계약일: NaiveDate::from_ymd(2015, 6, 2),
            물건금액: 0.0,
            건물면적: 80.0,
            토지면적: 100.0,
        },
        SeoulRealEstateTransactionPrice {
            계약일: NaiveDate::from_ymd(2015, 6, 2),
            물건금액: 0.0,
            건물면적: 160.0,
            토지면적: 200.0,
        },
    ] {
        let training_row = data.clone().to_training_row();
        println!(
            "{:?} 의 가격 예상가: {:?}만원",
            data,
            estimate(&training_row, &result) as usize
        );
    }

    Ok(())
}
