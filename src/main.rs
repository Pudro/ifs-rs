mod ifs;
mod runner;
use ifs::{Barnsley, Durer, IteratedFunctionSystem, Sierpinski};
mod params;
use runner::Runner;
mod operators;

fn main() {
    let target = IteratedFunctionSystem::barnsley_fern();

    let mut runner = Runner::new(target.clone());

    runner.generate_first_population();
    let mut last_best = 0.0;
    for i in 781..5001 {
        runner.step(i);
        println!("----------Generation {i}----------");
        println!("Mean Fitness: {:?}", runner.mean_fitness.unwrap());
        println!("Best Fitness: {:?}", runner.best.fitness);
        println!("Best IFS:{:#?}", runner.best);
        if runner.best.fitness > last_best {
            //runner.best.plot_fractal(500_000, (0.0, 0.0));
            runner.best.write_params(i);
            runner.plot_fitness_grid(i);
            last_best = runner.best.fitness;
        }
    }
}
