use crate::params::PARAMS;
use crate::runner::IFSBounds;
use plotters::prelude::*;
use rand::distributions::{Distribution, WeightedIndex};
use rand::seq::SliceRandom;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};

#[derive(Debug, Clone)]
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

pub trait Durer {
    fn durer_pentagon() -> Self;
}

impl IteratedFunctionSystem {
    pub fn new() -> Self {
        IteratedFunctionSystem {
            functions: Vec::new(),
            fitness: 0.0,
        }
    }

    pub fn add_function(&mut self, function: [f64; 6]) {
        self.functions.push(function);
    }

    pub fn apply_function(&self, point: (f64, f64)) -> (f64, f64) {
        let mut rng = rand::thread_rng();

        let determinants: Vec<f64> = self
            .functions
            .iter()
            .map(|f| {
                let result = (f[0] * f[3] - f[2] * f[1]).abs();
                if result > 0.0 {
                    result
                } else {
                    0.001
                }
            })
            .collect();

        let selection_probabilities: Vec<f64> = determinants
            .iter()
            .map(|d| d / determinants.iter().sum::<f64>())
            .collect();

        let weighted_index = WeightedIndex::new(selection_probabilities).unwrap();
        let selected_index = weighted_index.sample(&mut rng);
        let function = self.functions[selected_index].clone();

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

    pub fn generate_quantized_points(
        &self,
        num_points: i64,
        initial_point: (f64, f64),
        target_ifs_bounds: Option<&IFSBounds>,
    ) -> Vec<(f64, f64)> {
        let mut points: Vec<(f64, f64)> = vec![initial_point; num_points as usize];

        for i in 1..num_points as usize {
            points[i] = self.apply_function(points[i - 1])
        }

        let (lower_x, upper_x, lower_y, upper_y) = match target_ifs_bounds {
            Some(bounds) => {
                let lower_x = bounds.x_min;
                let upper_x = bounds.x_max;
                let lower_y = bounds.y_min;
                let upper_y = bounds.y_max;
                (lower_x, upper_x, lower_y, upper_y)
            }
            None => {
                let lower_x = points
                    .iter()
                    .map(|point| point.0)
                    .fold(f64::INFINITY, f64::min);

                let upper_x = points
                    .iter()
                    .map(|point| point.0)
                    .fold(f64::NEG_INFINITY, f64::max);

                let lower_y = points
                    .iter()
                    .map(|point| point.1)
                    .fold(f64::INFINITY, f64::min);

                let upper_y = points
                    .iter()
                    .map(|point| point.1)
                    .fold(f64::NEG_INFINITY, f64::max);
                (lower_x, upper_x, lower_y, upper_y)
            }
        };

        let x_range = upper_x - lower_x;
        let y_range = upper_y - lower_y;
        let x_resolution = (PARAMS.fitness_grid_resolution as f64 / x_range).ceil() as usize;
        let y_resolution = (PARAMS.fitness_grid_resolution as f64 / y_range).ceil() as usize;

        points = points
            .iter()
            .map(|(x, y)| {
                (
                    (x * x_resolution as f64).floor() / x_resolution as f64,
                    (y * y_resolution as f64).floor() / y_resolution as f64,
                )
            })
            .collect();

        points
    }

    pub fn plot_fractal(&self, num_points: i64, initial_point: (f64, f64)) {
        let points = self.generate_points(num_points, initial_point);

        let img_path = PARAMS.save_path.clone() + "/fractal.png";
        let root = BitMapBackend::new(&img_path, (2000, 2000)).into_drawing_area();
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

    pub fn plot_quantized_fractal(
        &self,
        num_points: i64,
        initial_point: (f64, f64),
        target_ifs_bounds: Option<&IFSBounds>,
    ) {
        let points = self.generate_quantized_points(num_points, initial_point, target_ifs_bounds);

        let img_path = PARAMS.save_path.clone() + "/fractal_quant.png";
        let root = BitMapBackend::new(&img_path, (2000, 2000)).into_drawing_area();
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

    pub fn write_params(&self, idx: usize) {
        let fname = PARAMS.save_path.clone() + &format!("/coeff_{idx}");
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(fname)
            .unwrap();

        let mut file = io::BufWriter::new(file);
        writeln!(file, "{:#?}", self).unwrap();
    }
}

impl Sierpinski for IteratedFunctionSystem {
    fn sierpinski_triangle() -> Self {
        let funcs: Vec<[f64; 6]> = vec![
            [0.5, 0., 0., 0.5, 0., 0.],
            [0.5, 0., 0., 0.5, 0.5, 0.],
            [0.5, 0., 0., 0.5, 0.25, 0.433],
        ];

        IteratedFunctionSystem {
            functions: funcs,
            fitness: 0.0,
        }
    }
}

impl Barnsley for IteratedFunctionSystem {
    fn barnsley_fern() -> Self {
        let funcs: Vec<[f64; 6]> = vec![
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

impl Durer for IteratedFunctionSystem {
    fn durer_pentagon() -> Self {
        let funcs: Vec<[f64; 6]> = vec![
            [0.382, 0., 0., 0.382, 0., 0.],
            [0.382, 0., 0., 0.382, 0.618, 0.],
            [0.382, 0., 0., 0.382, 0.809, 0.588],
            [0.382, 0., 0., 0.382, 0.309, 0.951],
            [0.382, 0., 0., 0.382, -0.191, 0.588],
            [-0.382, 0., 0., -0.382, 0.691, 0.951],
        ];

        IteratedFunctionSystem {
            functions: funcs,
            fitness: 0.0,
        }
    }
}
