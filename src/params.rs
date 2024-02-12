use config::{Config, ConfigBuilder, File};
use once_cell::sync::Lazy;
use std::env;
use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::sync::Mutex;

#[derive(Debug)]
pub struct Params {
    pub n_points: i64,
    pub initial_point: (f64, f64),
    pub min_individual_degree: i32,
    pub max_individual_degree: i32,
    pub initial_population_size: i32,
    pub recombination_population_size: i32,
    pub max_self_creation_population_size: i32,
    pub reassortment_population_size: i32,
    pub elite_fitness_threshold: f64,
    pub p_arithmetic_crossover: f64,
    pub p_vector_crossover: f64,
    pub a: f64,
    pub max_singel_coefficient: f64,
    pub min_singel_coefficient: f64,
    pub gaussian_mutation_radius: f64,
    pub mutation_probability: f64,
    pub fitness_grid_resolution: i32,
    pub p_rc: f64,
    pub p_ro: f64,
    pub p_at: f64,
    pub p_bd: f64,
    pub save_path: String,
}

pub static PARAMS: Lazy<Params> = Lazy::new(|| {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Provided args: {:#?}", args);
        eprintln!("Usage: {} <config_file>", args[0]);
        std::process::exit(1);
    }

    let config = Config::builder()
        .add_source(File::with_name(&args[1]))
        .build()
        .unwrap();

    if !Path::new(&config.get::<String>("save_path").unwrap()).exists() {
        fs::create_dir(config.get::<String>("save_path").unwrap()).unwrap();
    }

    let config_copy = config.get::<String>("save_path").unwrap() + "/" + &args[1];
    fs::copy(&args[1], config_copy).unwrap();

    Params {
        n_points: config.get("n_points").unwrap(),
        initial_point: config.get("initial_point").unwrap(),
        min_individual_degree: config.get("min_individual_degree").unwrap(),
        max_individual_degree: config.get("max_individual_degree").unwrap(),
        initial_population_size: config.get("initial_population_size").unwrap(),
        recombination_population_size: config.get("recombination_population_size").unwrap(),
        max_self_creation_population_size: config.get("max_self_creation_population_size").unwrap(),
        reassortment_population_size: config.get("reassortment_population_size").unwrap(),
        elite_fitness_threshold: config.get("elite_fitness_threshold").unwrap(),
        p_arithmetic_crossover: config.get("p_arithmetic_crossover").unwrap(),
        p_vector_crossover: config.get("p_vector_crossover").unwrap(),
        a: config.get("a").unwrap(),
        max_singel_coefficient: config.get("max_singel_coefficient").unwrap(),
        min_singel_coefficient: config.get("min_singel_coefficient").unwrap(),
        gaussian_mutation_radius: config.get("gaussian_mutation_radius").unwrap(),
        mutation_probability: config.get("mutation_probability").unwrap(),
        fitness_grid_resolution: config.get("fitness_grid_resolution").unwrap(),
        p_rc: config.get("p_rc").unwrap(),
        p_ro: config.get("p_ro").unwrap(),
        p_at: config.get("p_at").unwrap(),
        p_bd: config.get("p_bd").unwrap(),
        save_path: config.get("save_path").unwrap(),
    }
});
