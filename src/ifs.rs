use plotters::prelude::*;
use rand::seq::SliceRandom;

#[derive(Debug)]
pub struct IteratedFunctionSystem {
    pub functions: Vec<[f64; 6]>,
    pub fitness: f64,
}

pub trait Barnsley {
    fn barnsley_fern() -> Self;
}

pub trait Sierpinski {
    fn sierpinski_triangle() -> Self;
}

impl IteratedFunctionSystem {
    pub fn add_function(&mut self, function: [f64; 6]) {
        self.functions.push(function);
    }

    pub fn apply_function(&self, point: (f64, f64)) -> (f64, f64) {
        let mut rng = rand::thread_rng();
        let choice = self.functions.choose(&mut rng);
        let function = match choice {
            Some(function) => function,
            None => panic!("Could not draw a random function from the IFS!"),
        };

        (
            function[0] * point.0 + function[1] * point.1 + function[4],
            function[2] * point.0 + function[3] * point.1 + function[5],
        )
    }

    pub fn generate_points(&self, num_points: i64, initial_point: (f64, f64)) -> Vec<(f64, f64)> {
        let mut points: Vec<(f64, f64)> = vec![initial_point; num_points as usize];

        for i in 1..num_points as usize {
            points[i] = self.apply_function(points[i - 1])
        }

        points
    }

    pub fn plot_fractal(&self, num_points: i64, initial_point: (f64, f64)) {
        let points = self.generate_points(num_points, initial_point);

        let root = BitMapBackend::new("fractal.png", (800, 800)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .margin(5)
            .build_cartesian_2d(-2.0..2.0, -0.5..8.0)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("x")
            .y_desc("y")
            .draw()
            .unwrap();

        chart
            .draw_series(
                points
                    .iter()
                    .map(|point| Circle::new((point.0, point.1), 1, BLACK.filled())),
            )
            .unwrap();
    }
}

impl Barnsley for IteratedFunctionSystem {
    fn barnsley_fern() -> Self {
        let funcs: Vec<[f64; 6]> = vec![
            [0., 0., 0., 0.16, 0., 0.],
            [0., 0., 0., 0.16, 0., 0.0],
            [0.85, 0.04, -0.04, 0.85, 0.0, 1.6],
            [0.2, -0.26, 0.23, 0.22, 0.0, 1.6],
            [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44],
        ];

        IteratedFunctionSystem {
            functions: funcs,
            fitness: 0.0,
        }
    }
}
