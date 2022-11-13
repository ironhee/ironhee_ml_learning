use plotters::prelude::*;

pub fn draw_costs_plot(
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
