mod ifs;
mod runner;
use std::process::exit;

use ifs::{Barnsley, IteratedFunctionSystem, Sierpinski};
mod params;
use params::PARAMS;
use runner::Runner;
mod operators;
use crate::operators::{arithmetic_crossover, gaussian_mutation, reassortment};
use operators::one_point_crossover;

// ifs.add_function(np.array([0, 0, 0, 0.16, 0, 0.0]))
// ifs.add_function(np.array([0.85, 0.04, -0.04, 0.85, 0.0, 1.6]))
// ifs.add_function(np.array([0.2, -0.26, 0.23, 0.22, 0.0, 1.6]))
// ifs.add_function(np.array([-0.15, 0.28, 0.26, 0.24, 0.0, 0.44]))
fn main() {
    let target = IteratedFunctionSystem::barnsley_fern();

    let mut runner = Runner::new(target);
    runner.generate_first_population();

    for i in 0..4001 {
        runner.step();
        println!("----------Generation {i}----------");
        println!("Mean Fitness: {:?}", runner.mean_fitness.unwrap());
        println!("Best Fitness: {:?}", runner.best.fitness);
        println!("Best IFS:{:#?}", runner.best);
        if i % 100 == 0 {
            runner.best.plot_fractal(500_000, (0.0, 0.0));
            runner.best.write_params(i);
            runner.plot_fitness_grid(i);
        }
    }
}
