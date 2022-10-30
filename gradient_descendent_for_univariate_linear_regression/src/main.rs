use num_traits::pow;
use plotters::prelude::*;
use rand::Rng;

#[derive(Debug)]
struct TrainingRow {
    feature: f64,
    target: f64,
}

#[derive(Debug, Clone, Copy)]
struct EstimationModel {
    slope: f64,
    intercept: f64,
}

#[derive(Debug)]
struct Gradient {
    slope: f64,
    intercept: f64,
}

fn estimate(training_row: &TrainingRow, estimation_model: &EstimationModel) -> f64 {
    estimation_model.slope * training_row.feature + estimation_model.intercept
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
    let mut slope_gratient = 0.0;
    let mut intercept_gratient = 0.0;

    for training_row in training_rows {
        let cost_of_row = get_cost_of_row(training_row, estimation_model);
        slope_gratient += cost_of_row * training_row.feature;
        intercept_gratient += cost_of_row;
    }
    slope_gratient /= row_count as f64;
    intercept_gratient /= row_count as f64;

    Gradient {
        slope: slope_gratient,
        intercept: intercept_gratient,
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
            slope: estimation_model.slope - learning_rate * gradient.slope,
            intercept: estimation_model.intercept - learning_rate * gradient.intercept,
        }
    }

    (estimation_model, costs)
}

fn draw_costs_plot(
    filename: &str,
    learning_rate: f64,
    costs: &[f64],
    iteraction_start: usize,
    iteration_end: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Plot
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(
            format!(
                "Learning Rate: {}, iteration [{}..{}]",
                learning_rate, iteraction_start, iteration_end
            ),
            ("sans-serif", 20),
        )
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(
            iteraction_start..iteration_end,
            0.0..(costs[iteraction_start] * 1.2),
        )?;
    chart
        .configure_mesh()
        .x_desc("iteration")
        .y_desc("cost of function")
        .draw()?;
    chart.draw_series(LineSeries::new(
        costs[iteraction_start..iteration_end]
            .iter()
            .enumerate()
            .map(|(index, cost)| (iteraction_start + index, cost.clone())),
        &RED,
    ))?;
    chart.configure_series_labels().draw()?;
    root.present()?;
    Ok(())
}

fn draw_model_plot(
    filename: &str,
    training_rows: &[TrainingRow],
    estimation_model: &EstimationModel,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            f64::from(0.0)..f64::from(100.0),
            f64::from(0.0)..f64::from(100.0),
        )?;
    chart
        .configure_mesh()
        .x_desc("feature")
        .y_desc("target")
        .draw()?;
    chart
        .draw_series(
            training_rows
                .iter()
                .map(|root| Circle::new((root.feature, root.target), 2, GREEN.filled())),
        )?
        .label("target")
        .legend(|(x, y)| Circle::new((x + 10, y), 2, GREEN.filled()));
    chart
        .draw_series(LineSeries::new(
            vec![
                (0.0, estimation_model.intercept),
                (100.0, 100.0 * estimation_model.slope),
            ],
            &RED,
        ))?
        .label("estimation")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    root.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    let training_rows: Vec<TrainingRow> = (0..100)
        .into_iter()
        .map(|x| TrainingRow {
            feature: f64::from(x),
            target: f64::from(x) + 1.0 + rng.gen_range(-10.0..10.0),
        })
        .collect();

    let learning_rate = 1.0e-5;
    let iteration_count = 1000;
    let initial = EstimationModel {
        slope: 0.0,
        intercept: 0.0,
    };

    let (result, costs) =
        gradient_descent(&training_rows, &initial, learning_rate, iteration_count);

    draw_model_plot("plot/before_training.png", &training_rows, &initial)?;
    draw_model_plot("plot/after_training.png", &training_rows, &result)?;

    draw_costs_plot("plot/training.png", learning_rate, &costs, 0, 100)?;
    Ok(())
}
