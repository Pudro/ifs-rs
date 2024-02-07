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
    let mut ifs = IteratedFunctionSystem::new();
    //ifs.add_function([0.0, 0.0, 0.0, 0.18, 0.0, 0.0]);
    //ifs.add_function([0.85, -0.02, -0.04, 0.85, 0.0, 1.6]);
    //ifs.add_function([0.2, -0.26, 0.23, 0.26, 0.8, 1.6]);
    //ifs.add_function([-0.15, 0.48, 0.26, 0.24, 0.0, 0.44]);
    ifs.add_function([
        -0.14079454891839838,
        -0.01544179707707128,
        1.0449283271917316,
        0.7240550936175718,
        -0.005777335407057577,
        1.208236280416499,
    ]);
    ifs.add_function([
        -0.9911183315021542,
        0.07357349551068212,
        1.029947233420324,
        0.5124350411702581,
        0.0955382729113493,
        1.1090597987568371,
    ]);
    ifs.add_function([
        -0.7106331369436815,
        0.15176695746077185,
        1.0138001270850747,
        0.8715580792454672,
        0.056855408826728446,
        1.0401209995629992,
    ]);
    ifs.add_function([
        -0.9574993147926969,
        -0.04619177781525098,
        1.1317340615395093,
        0.4049061874158196,
        0.018004630500959504,
        0.8944125531225038,
    ]);
    runner.best = ifs.clone();
    runner.population.push(ifs.clone());
    runner.step();
    for _ in 0..5 {
        runner.population.push(ifs.clone());
    }
    runner.plot_fitness_grid(0);
    runner.best.plot_fractal(500_000, (0.0, 0.0));

    for i in 1..2001 {
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
