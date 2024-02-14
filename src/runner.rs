use linreg::linear_regression_of;
use plotters::prelude::*;
use plotters::style::full_palette::{BLUE_300, GREEN_900, GREY_100, GREY_50, PURPLE_300};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use rand_distr::Normal;
use std::collections::HashSet;
use std::fmt::format;
use std::hash::Hash;

use crate::{ifs::IteratedFunctionSystem, params::PARAMS};
//use kiddo::float::kdtree::KdTree;
//use kiddo::KdTree;
//use kiddo::NearestNeighbour;
//use kiddo::SquaredEuclidean;
use crate::operators::{
    arithmetic_crossover, gaussian_mutation, one_point_crossover, random_or_binary_mutation,
    reassortment,
};
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use ndarray::prelude::*;
use ordered_float::OrderedFloat;

#[derive(Debug)]
pub struct Runner {
    pub population: Vec<IteratedFunctionSystem>,
    pub target_ifs: IteratedFunctionSystem,
    pub target_ifs_points: Vec<(f64, f64)>,
    target_ifs_bd: f64,
    _cached_ifs_bounds: Option<IFSBounds>,
    pub mean_fitness: Option<f64>,
    pub best: IteratedFunctionSystem,
    _kdtree: Option<KdTree<f64, i32, [f64; 2]>>,
    _grid_points: Vec<[f64; 2]>,
}

#[derive(Debug)]
pub struct IFSBounds {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl Runner {
    pub fn new(target_ifs: IteratedFunctionSystem) -> Self {
        // let target_ifs_points = target_ifs.generate_points(PARAMS.n_points, PARAMS.initial_point);

        // very ugly hot fix
        let mut dummy_runner = Runner {
            population: Vec::new(),
            target_ifs: target_ifs.clone(),
            target_ifs_points: target_ifs.generate_points(PARAMS.n_points, PARAMS.initial_point),
            target_ifs_bd: 0.0,
            _cached_ifs_bounds: None,
            mean_fitness: None,
            best: IteratedFunctionSystem::new(),
            _kdtree: None,
            _grid_points: Vec::new(),
        };

        let target_ifs_points = target_ifs.generate_quantized_points(
            PARAMS.n_points,
            PARAMS.initial_point,
            Some(dummy_runner.target_ifs_bounds()),
        );

        let target_ifs_bd = dummy_runner._calculate_box_dimension(target_ifs_points.clone());

        Runner {
            population: Vec::new(),
            target_ifs,
            target_ifs_points,
            target_ifs_bd,
            _cached_ifs_bounds: None,
            mean_fitness: None,
            best: IteratedFunctionSystem::new(),
            _kdtree: None,
            _grid_points: Vec::new(),
        }
    }

    pub fn target_ifs_bounds(&mut self) -> &IFSBounds {
        if self._cached_ifs_bounds.is_none() {
            let lower_x = self
                .target_ifs_points
                .iter()
                .map(|point| point.0)
                .fold(f64::INFINITY, f64::min);

            let upper_x = self
                .target_ifs_points
                .iter()
                .map(|point| point.0)
                .fold(f64::NEG_INFINITY, f64::max);

            let lower_y = self
                .target_ifs_points
                .iter()
                .map(|point| point.1)
                .fold(f64::INFINITY, f64::min);

            let upper_y = self
                .target_ifs_points
                .iter()
                .map(|point| point.1)
                .fold(f64::NEG_INFINITY, f64::max);

            let bounds = IFSBounds {
                x_min: lower_x,
                x_max: upper_x,
                y_min: lower_y,
                y_max: upper_y,
            };
            self._cached_ifs_bounds = Some(bounds);
        }

        self._cached_ifs_bounds.as_ref().unwrap()
    }

    pub fn calculate_fitness(&mut self, attractor_ifs: &IteratedFunctionSystem) -> f64 {
        let attractor_points = attractor_ifs.generate_points(PARAMS.n_points, PARAMS.initial_point);

        if self._kdtree.is_none() {
            let grid_x = Array::linspace(
                self.target_ifs_bounds().x_min,
                self.target_ifs_bounds().x_max,
                PARAMS.fitness_grid_resolution as usize,
            );
            let grid_y = Array::linspace(
                self.target_ifs_bounds().y_min,
                self.target_ifs_bounds().y_max,
                PARAMS.fitness_grid_resolution as usize,
            );

            self._grid_points = Vec::new();
            for &x in grid_x.iter() {
                for &y in grid_y.iter() {
                    self._grid_points.push([x, y]);
                }
            }

            let mut kdtree: KdTree<f64, i32, [f64; 2]> = KdTree::new(2);

            for (i, point) in self._grid_points.iter().enumerate() {
                kdtree.add(*point, i as i32).unwrap();
            }

            self._kdtree = Some(kdtree);
        }

        let attractor_points_inside_bounds: Vec<[f64; 2]> = attractor_points
            .iter()
            .filter(|&&(x, y)| {
                x >= self.target_ifs_bounds().x_min
                    && x <= self.target_ifs_bounds().x_max
                    && y >= self.target_ifs_bounds().y_min
                    && y <= self.target_ifs_bounds().y_max
            })
            .map(|&(x, y)| [x, y])
            .collect::<Vec<[f64; 2]>>();

        let target_closest_points = self
            .target_ifs_points
            .iter()
            .map(|(x, y)| {
                self._grid_points[*self
                    ._kdtree
                    .as_ref()
                    .unwrap()
                    .nearest(&[*x, *y], 1, &squared_euclidean)
                    .unwrap()[0]
                    .1 as usize]
            })
            .collect::<Vec<[f64; 2]>>();

        let attractor_closest_points = attractor_points_inside_bounds
            .iter()
            .map(|point| {
                self._grid_points[*self
                    ._kdtree
                    .as_ref()
                    .unwrap()
                    .nearest(point, 1, &squared_euclidean)
                    .unwrap()[0]
                    .1 as usize]
            })
            .collect::<Vec<[f64; 2]>>();

        let attractor_set: HashSet<_> = attractor_closest_points
            .iter()
            .cloned()
            .map(|point| [OrderedFloat(point[0]), OrderedFloat(point[1])])
            .collect();

        let target_set: HashSet<_> = target_closest_points
            .iter()
            .cloned()
            .map(|point| [OrderedFloat(point[0]), OrderedFloat(point[1])])
            .collect();

        let n_nn = attractor_set.difference(&target_set).count();
        let n_nd = target_set.difference(&attractor_set).count();

        let n_a = attractor_set.len();
        let n_i = target_set.len();

        let r_c = n_nd as f64 / n_i as f64;
        let r_o = n_nn as f64 / n_a as f64;

        let fitness = (PARAMS.p_rc * (1.0 - r_c)
            + PARAMS.p_ro * (1.0 - r_o)
            + PARAMS.p_at * (1.0 - (n_a.abs_diff(n_i) as f64) / (n_a as f64 + n_i as f64)))
            / (PARAMS.p_rc + PARAMS.p_ro + PARAMS.p_at);

        //println!("n_nn:{:?}", n_nn);
        //println!("n_a:{:?}", n_a);
        //println!("n_nd:{:?}", n_nd);
        //println!("n_i:{:?}", n_i);
        //println!("r_c:{:?}", r_c);
        //println!("r_o:{:?}", r_o);
        //println!("fitness:{:?}", fitness);
        //println!("----");

        fitness
    }

    pub fn calculate_fitness_from_quantized(
        &mut self,
        attractor_ifs: &IteratedFunctionSystem,
    ) -> f64 {
        let attractor_points = attractor_ifs.generate_quantized_points(
            PARAMS.n_points,
            PARAMS.initial_point,
            Some(self.target_ifs_bounds()),
        );

        //self._count_boxes(attractor_points.clone());
        //let bd_a = self._calculate_box_dimension(attractor_points.clone());
        //let bd_t = self.target_ifs_bd;

        let target_closest_points = self.target_ifs_points.clone();

        let attractor_closest_points = attractor_points;

        let attractor_set: HashSet<_> = attractor_closest_points
            .iter()
            .cloned()
            .map(|point| [OrderedFloat(point.0), OrderedFloat(point.1)])
            .collect();

        let target_set: HashSet<_> = target_closest_points
            .iter()
            .cloned()
            .map(|point| [OrderedFloat(point.0), OrderedFloat(point.1)])
            .collect();

        let n_nn = attractor_set.difference(&target_set).count();
        let n_nd = target_set.difference(&attractor_set).count();

        let n_a = attractor_set.len();
        let n_i = target_set.len();

        let r_c = n_nd as f64 / n_i as f64;
        let r_o = n_nn as f64 / n_a as f64;

        let fitness = (PARAMS.p_rc * (1.0 - r_c)
            + PARAMS.p_ro * (1.0 - r_o)
            + PARAMS.p_at * (1.0 - (n_a.abs_diff(n_i) as f64) / (n_a as f64 + n_i as f64)))
            //+ PARAMS.p_bd * (1.0 - (bd_a - bd_t).abs() * 10.0 / bd_t).max(0.0))
            / (PARAMS.p_rc + PARAMS.p_ro + PARAMS.p_at + PARAMS.p_bd);

        //println!("box: {:#?}", (bd_a - bd_t).abs() / bd_t);
        //println!(
        //    "bd: {:#?}",
        //    PARAMS.p_bd * (1.0 - (bd_a - bd_t).abs() / bd_t)
        //);
        // println!(
        //     "at: {:#?}",
        //     PARAMS.p_at * (1.0 - (n_a.abs_diff(n_i) as f64) / (n_a as f64 + n_i as f64))
        // );
        // println!("ro: {:#?}", PARAMS.p_ro * (1.0 - r_o));
        // println!("rc: {:#?}", PARAMS.p_rc * (1.0 - r_c));
        //println!("n_nn:{:?}", n_nn);
        //println!("n_a:{:?}", n_a);
        //println!("n_nd:{:?}", n_nd);
        //println!("n_i:{:?}", n_i);
        //println!("r_c:{:?}", r_c);
        //println!("r_o:{:?}", r_o);
        //println!("fitness:{:?}", fitness);
        //println!("----");

        fitness
    }

    pub fn step(&mut self, i: usize) {
        let mut rng = rand::thread_rng();
        let mut new_pop = Vec::new();

        for idx in 0..self.population.len() {
            let ifs = &self.population[idx].clone();
            if self.population[idx].fitness == 0.0 {
                self.population[idx].fitness = self.calculate_fitness_from_quantized(ifs);
            }
        }

        self.mean_fitness = Some(
            self.population.iter().map(|i| i.fitness).sum::<f64>() / self.population.len() as f64,
        );

        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // =========================
        self.plot_comparison(self.population[1].clone(), i);
        // =========================

        if self.population[0].fitness > self.best.fitness {
            self.best = self.population[0].clone();
        }

        new_pop.extend(self.recombination_offspring());
        new_pop.extend(self.self_creation_offspring());
        new_pop.extend(self.reassortment_offspring());
        new_pop.extend(
            self.population[..(PARAMS.recombination_population_size
                + PARAMS.reassortment_population_size
                + PARAMS.max_self_creation_population_size) as usize
                / 10]
                .iter()
                .cloned(),
        );

        for ifs in new_pop.iter_mut() {
            if rng.gen::<f64>() < self.mean_fitness.unwrap() {
                gaussian_mutation(ifs);
            } else {
                random_or_binary_mutation(ifs);
            }
        }

        if self.best.fitness > PARAMS.elite_fitness_threshold {
            new_pop.push(self.best.clone());
        }

        self.population = new_pop;
    }

    pub fn generate_first_population(&mut self) {
        let mut rng = rand::thread_rng();

        for _ in 0..PARAMS.initial_population_size {
            let size =
                rng.gen_range(PARAMS.min_individual_degree..PARAMS.max_individual_degree + 1);
            let mut ifs = IteratedFunctionSystem::new();

            for _ in 0..size {
                let random_array: [f64; 6] = rng.gen();
                ifs.add_function(random_array);
            }

            self.population.push(ifs);
        }
    }

    pub fn fitness_proportional_selection(&mut self) -> Vec<IteratedFunctionSystem> {
        let total_fitness = self
            .population
            .iter()
            .map(|individual| individual.fitness)
            .sum::<f64>();

        let selection_probabilities = self
            .population
            .iter()
            .map(|individual| individual.fitness / total_fitness)
            .collect::<Vec<f64>>();

        let weighted_index = WeightedIndex::new(selection_probabilities).unwrap();
        let mut rng = rand::thread_rng();
        let population = self.population.clone();
        let mut selected_individuals = Vec::new();

        let mut selected_indexes = HashSet::with_capacity(2);
        while selected_indexes.len() < 2 {
            let selected_index = weighted_index.sample(&mut rng);
            selected_indexes.insert(selected_index);
        }

        for idx in selected_indexes.iter() {
            selected_individuals.push(population[*idx].clone());
        }

        selected_individuals
    }

    pub fn recombination_offspring(&mut self) -> Vec<IteratedFunctionSystem> {
        let mut offspring = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..(PARAMS.recombination_population_size / 2) {
            let parents = self.fitness_proportional_selection();
            if rng.gen::<f64>() <= PARAMS.p_arithmetic_crossover {
                let (c1, c2) = arithmetic_crossover(&parents[0], &parents[1]);
                offspring.push(c1);
                offspring.push(c2);
            } else {
                let (c1, c2) = one_point_crossover(&parents[0], &parents[1]);
                offspring.push(c1);
                offspring.push(c2);
            }
        }
        offspring
    }

    pub fn self_creation_offspring(&mut self) -> Vec<IteratedFunctionSystem> {
        let mut offspring = Vec::new();
        let mut rng = rand::thread_rng();
        let n_specimen;

        if 1.0 / self.mean_fitness.unwrap() > PARAMS.max_self_creation_population_size as f64 {
            n_specimen = PARAMS.max_self_creation_population_size;
        } else {
            n_specimen = (1.0 / self.mean_fitness.unwrap()) as i32;
        }

        let genetic_universum: Vec<[f64; 6]> = self
            .population
            .iter()
            .flat_map(|ifs| ifs.functions.iter())
            .cloned()
            .collect();

        let size = rand::thread_rng()
            .gen_range(PARAMS.min_individual_degree..=PARAMS.max_individual_degree);

        for _ in 0..n_specimen {
            let mut ifs = IteratedFunctionSystem::new();
            ifs.functions
                .extend(genetic_universum.choose_multiple(&mut rng, size as usize));
            offspring.push(ifs);
        }

        offspring
    }

    pub fn reassortment_offspring(&mut self) -> Vec<IteratedFunctionSystem> {
        let mut offspring = Vec::new();

        for _ in 0..(PARAMS.reassortment_population_size / 2) {
            let parents = self.fitness_proportional_selection();
            let (c1, c2) = reassortment(&parents[0], &parents[1]);
            offspring.push(c1);
            offspring.push(c2);
        }

        offspring
    }

    pub fn plot_fitness_grid(&mut self, iter: usize) {
        let attractor_ifs = self.best.clone();
        let attractor_points = attractor_ifs.generate_points(PARAMS.n_points, PARAMS.initial_point);
        //let attractor_points_inside_bounds: Vec<[f64; 2]> = attractor_points
        //    .iter()
        //    .filter(|&&(x, y)| {
        //        x >= self.target_ifs_bounds().x_min
        //            && x <= self.target_ifs_bounds().x_max
        //            && y >= self.target_ifs_bounds().y_min
        //            && y <= self.target_ifs_bounds().y_max
        //    })
        //    .map(|&(x, y)| [x, y])
        //    .collect::<Vec<[f64; 2]>>();

        let target_points = self
            .target_ifs
            .generate_points(PARAMS.n_points, PARAMS.initial_point);

        //let target_closest_points = target_points
        //    .iter()
        //    .map(|(x, y)| {
        //        self._grid_points[*self
        //            ._kdtree
        //            .as_ref()
        //            .unwrap()
        //            .nearest(&[*x, *y], 1, &squared_euclidean)
        //            .unwrap()[0]
        //            .1 as usize]
        //    })
        //    .collect::<Vec<[f64; 2]>>();

        //let attractor_closest_points = attractor_points_inside_bounds
        //    .iter()
        //    .map(|point| {
        //        self._grid_points[*self
        //            ._kdtree
        //            .as_ref()
        //            .unwrap()
        //            .nearest(point, 1, &squared_euclidean)
        //            .unwrap()[0]
        //            .1 as usize]
        //    })
        //    .collect::<Vec<[f64; 2]>>();

        //let attractor_set: HashSet<_> = attractor_closest_points
        //    .iter()
        //    .cloned()
        //    .map(|point| [OrderedFloat(point[0]), OrderedFloat(point[1])])
        //    .collect();

        //let target_set: HashSet<_> = target_closest_points
        //    .iter()
        //    .cloned()
        //    .map(|point| [OrderedFloat(point[0]), OrderedFloat(point[1])])
        //    .collect();

        let img_name = format!("/{}.png", iter);
        let img_path = PARAMS.save_path.clone() + &img_name;
        let root = BitMapBackend::new(&img_path, (1200, 1200)).into_drawing_area();

        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .build_cartesian_2d(
                self.target_ifs_bounds().x_min..self.target_ifs_bounds().x_max,
                self.target_ifs_bounds().y_min..self.target_ifs_bounds().y_max,
            )
            .unwrap();

        //chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(
                self._grid_points
                    .iter()
                    .map(|point| Circle::new((point[0], point[1]), 1, GREY_100.filled())),
            )
            .unwrap();

        chart
            .draw_series(
                target_points
                    .iter()
                    .map(|point| Circle::new((point.0, point.1), 1, BLUE.filled())),
            )
            .unwrap()
            .label("Target Points")
            .legend(|(x, y)| Rectangle::new([(x, y), (x + 10, y + 5)], BLUE.filled()));

        chart
            .draw_series(
                attractor_points
                    .iter()
                    .map(|point| Circle::new((point.0, point.1), 1, RED.filled())),
            )
            .unwrap()
            .label("Attractor Points")
            .legend(|(x, y)| Rectangle::new([(x, y), (x + 10, y + 5)], RED.filled()));

        //chart
        //    .draw_series(target_set.difference(&attractor_set).cloned().map(|point| {
        //        Circle::new(
        //            (f64::from(point[0]), f64::from(point[1])),
        //            1.5,
        //            BLUE_300.filled(),
        //        )
        //    }))
        //    .unwrap()
        //    .label("Points not drawn")
        //    .legend(|(x, y)| Rectangle::new([(x, y), (x + 10, y + 5)], BLUE_300.filled()));

        //chart
        //    .draw_series(attractor_set.difference(&target_set).cloned().map(|point| {
        //        Circle::new(
        //            (f64::from(point[0]), f64::from(point[1])),
        //            1.5,
        //            PURPLE_300.filled(),
        //        )
        //    }))
        //    .unwrap()
        //    .label("Points not needed")
        //    .legend(|(x, y)| Rectangle::new([(x, y), (x + 10, y + 5)], PURPLE_300.filled()));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.5))
            .position(SeriesLabelPosition::UpperLeft)
            .label_font(("Calibri", 26))
            .draw()
            .unwrap();
    }

    fn _count_boxes(&self, attractor_points: Vec<(f64, f64)>) {
        let x_min = attractor_points
            .iter()
            .map(|&(x, _)| x)
            .fold(f64::INFINITY, f64::min);

        let x_max = attractor_points
            .iter()
            .map(|&(x, _)| x)
            .fold(f64::NEG_INFINITY, f64::max);

        let y_min = attractor_points
            .iter()
            .map(|&(_, y)| y)
            .fold(f64::INFINITY, f64::min);

        let y_max = attractor_points
            .iter()
            .map(|&(_, y)| y)
            .fold(f64::NEG_INFINITY, f64::max);

        let lx = (x_max - x_min).abs();
        let ly = (y_max - y_min).abs();

        println!("{} {}", lx, ly);

        // computing the fractal dimension
        // considering only scales in a logarithmic list
        let mut scales = Vec::new();
        for i in 0..10 {
            let scale = 2_f64.powf(i as f64 * 0.01);
            println!("======= Scale : {:?}", scale);
            scales.push(scale);
        }

        let mut ns = Vec::new();
        // looping over several scales
        for &scale in &scales {
            // computing the histogram
            let mut hist =
                vec![vec![0; (lx as f64 / scale) as usize]; (ly as f64 / scale) as usize];
            for &(x, y) in &attractor_points {
                println!("scale:{:#?}", scale);
                let x_index = (x as f64 / scale) as usize;
                let y_index = (y as f64 / scale) as usize;
                hist[x_index][y_index] += 1;
            }
            let count = hist.iter().flatten().filter(|&&val| val > 0).count();
            ns.push(count);
        }

        // linear fit, polynomial of degree 1
        let mut x_log = Vec::new();
        let mut ns_log = Vec::new();
        for (&scale, &count) in scales.iter().zip(&ns) {
            x_log.push(scale.log2());
            ns_log.push(count as f64);
        }

        let sum_x_log = x_log.iter().sum::<f64>();
        let sum_ns_log = ns_log.iter().sum::<f64>();
        let sum_x_log_squared = x_log.iter().map(|&x| x * x).sum::<f64>();
        let sum_x_log_ns_log = x_log
            .iter()
            .zip(&ns_log)
            .map(|(&x, &ns)| x * ns)
            .sum::<f64>();

        let n = scales.len() as f64; // Number of scales used
        let slope = (n * sum_x_log_ns_log - sum_x_log * sum_ns_log)
            / (n * sum_x_log_squared - sum_x_log * sum_x_log);
        println!("{:?}", slope);
    }

    pub fn plot_comparison(&mut self, ifs: IteratedFunctionSystem, iter: usize) {
        let attractor_ifs = ifs;
        let attractor_points = attractor_ifs.generate_points(PARAMS.n_points, PARAMS.initial_point);
        let target_points = self
            .target_ifs
            .generate_points(PARAMS.n_points, PARAMS.initial_point);
        let img_name = format!("/gen_{}.png", iter);
        let img_path = PARAMS.save_path.clone() + &img_name;
        let root = BitMapBackend::new(&img_path, (1200, 1200)).into_drawing_area();

        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .build_cartesian_2d(
                self.target_ifs_bounds().x_min..self.target_ifs_bounds().x_max,
                self.target_ifs_bounds().y_min..self.target_ifs_bounds().y_max,
            )
            .unwrap();

        chart
            .draw_series(
                self._grid_points
                    .iter()
                    .map(|point| Circle::new((point[0], point[1]), 1, GREY_100.filled())),
            )
            .unwrap();

        chart
            .draw_series(
                target_points
                    .iter()
                    .map(|point| Circle::new((point.0, point.1), 1, BLUE.filled())),
            )
            .unwrap()
            .label("Target Points")
            .legend(|(x, y)| Rectangle::new([(x, y), (x + 10, y + 5)], BLUE.filled()));

        chart
            .draw_series(
                attractor_points
                    .iter()
                    .map(|point| Circle::new((point.0, point.1), 1, RED.filled())),
            )
            .unwrap()
            .label("Attractor Points")
            .legend(|(x, y)| Rectangle::new([(x, y), (x + 10, y + 5)], RED.filled()));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.5))
            .position(SeriesLabelPosition::UpperLeft)
            .label_font(("Calibri", 26))
            .draw()
            .unwrap();
    }

    pub fn _calculate_box_dimension(&self, attractor_points: Vec<(f64, f64)>) -> f64 {
        let x_min = attractor_points
            .iter()
            .map(|&(x, _)| x)
            .fold(f64::INFINITY, f64::min);

        let x_max = attractor_points
            .iter()
            .map(|&(x, _)| x)
            .fold(f64::NEG_INFINITY, f64::max);

        let y_min = attractor_points
            .iter()
            .map(|&(_, y)| y)
            .fold(f64::INFINITY, f64::min);

        let y_max = attractor_points
            .iter()
            .map(|&(_, y)| y)
            .fold(f64::NEG_INFINITY, f64::max);

        //let scales = vec![15., 45.254834, 64., 90.50966799, 128., 181.01933598];
        let scales = vec![50., 100., 150., 250., 300., 350.];
        //let scales = vec![5.];

        let mut dimensions = Vec::new();

        for scale in scales {
            let box_size_x = (x_max - x_min) / scale;
            let box_size_y = (y_max - y_min) / scale;
            let mut boxes = HashSet::new();

            //println!("box_size_x,y: {:?}, {:?}", box_size_x, box_size_y);
            for &(x, y) in attractor_points.iter() {
                let box_x = (x / box_size_x).floor() as i32;
                let box_y = (y / box_size_y).floor() as i32;

                boxes.insert((box_x, box_y));
            }

            let mut edge_boxes = HashSet::new();
            for &b in boxes.iter() {
                let neighbors = [
                    (b.0 - 1, b.1),
                    (b.0 + 1, b.1),
                    (b.0, b.1 - 1),
                    (b.0, b.1 + 1),
                ];

                let mut all_neighbors_in_boxes = true;
                for neighbor in neighbors.iter() {
                    if !boxes.contains(neighbor) {
                        all_neighbors_in_boxes = false;
                        break;
                    }
                }

                if !all_neighbors_in_boxes {
                    edge_boxes.insert(b);
                }
                //check if all neighbors in boxes;
                //if not, insert b into edge_boxes
            }

            dimensions.push((scale, boxes.len() as f64));
            //println!("boxes: {:?}", boxes);
            //println!("boxes len: {:?}", boxes.len());

            //println!("edge_boxes: {:?}", edge_boxes);
            //println!("edge_boxes len: {:?}", edge_boxes.len());
        }

        let mut log_scales = Vec::new();
        let mut log_counts = Vec::new();

        for (scale, count) in dimensions {
            log_scales.push(scale);
            log_counts.push(count);
        }

        let log_log = log_scales
            .iter()
            .zip(log_counts.iter())
            .map(|(&x, &y)| (x.ln(), y.ln()))
            .collect::<Vec<_>>();

        let n = log_log.len() as f64;
        let sum_xy = log_log.iter().fold(0.0, |acc, &(x, y)| acc + x * y);
        let sum_x = log_log.iter().fold(0.0, |acc, &(x, _)| acc + x);
        let sum_y = log_log.iter().fold(0.0, |acc, &(_, y)| acc + y);
        let sum_x_squared = log_log.iter().fold(0.0, |acc, &(x, _)| acc + x * x);

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x.powi(2));

        //println!("slope: {:#?}", slope);
        slope
    }
}
