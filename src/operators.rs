use crate::ifs::IteratedFunctionSystem;
use crate::params::PARAMS;
use rand::{distributions::Uniform, seq::SliceRandom, thread_rng, Rng};
use rand_distr::Normal;

pub fn arithmetic_crossover(
    p1: &IteratedFunctionSystem,
    p2: &IteratedFunctionSystem,
) -> (IteratedFunctionSystem, IteratedFunctionSystem) {
    let mut functions1: Vec<[f64; 6]> = Vec::new();
    let mut functions2: Vec<[f64; 6]> = Vec::new();

    for (f1, f2) in p1.functions.iter().zip(p2.functions.iter()) {
        let nf1 = f1
            .iter()
            .zip(f2.iter())
            .map(|(e1, e2)| e1 * (1.0 - PARAMS.a) + e2 * PARAMS.a)
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap();
        functions1.push(nf1);

        let nf2 = f1
            .iter()
            .zip(f2.iter())
            .map(|(e1, e2)| e1 * PARAMS.a + e2 * (1.0 - PARAMS.a))
            .collect::<Vec<f64>>()
            .try_into()
            .unwrap();
        functions2.push(nf2);
    }
    let c1 = IteratedFunctionSystem {
        functions: functions1,
        fitness: 0.0,
    };
    let c2 = IteratedFunctionSystem {
        functions: functions2,
        fitness: 0.0,
    };

    return (c1, c2);
}

// rewrite this to operate on functions instead of IFSs
pub fn one_point_crossover(
    p1: &IteratedFunctionSystem,
    p2: &IteratedFunctionSystem,
) -> (IteratedFunctionSystem, IteratedFunctionSystem) {
    let crossover_point = rand::thread_rng().gen_range(0..p1.functions[0].len());

    let mut c1 = IteratedFunctionSystem {
        functions: Vec::new(),
        fitness: 0.0,
    };

    let mut c2 = IteratedFunctionSystem {
        functions: Vec::new(),
        fitness: 0.0,
    };

    for (f1, f2) in p1.functions.iter().zip(p2.functions.iter()) {
        if rand::thread_rng().gen::<f64>() < 0.5 {
            let function1: [f64; 6] = f1[..crossover_point]
                .iter()
                .copied()
                .chain(f2[crossover_point..].iter().copied())
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();
            let function2: [f64; 6] = f1[..crossover_point]
                .iter()
                .copied()
                .chain(f2[crossover_point..].iter().copied())
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();
            c1.functions.push(function1);
            c2.functions.push(function2);
        } else {
            c1.functions.push(*f1);
            c2.functions.push(*f2);
        }
    }

    c1.functions
        .extend_from_slice(&p2.functions[crossover_point..]);

    let mut c2 = IteratedFunctionSystem {
        functions: p2.functions[..crossover_point].to_vec(),
        fitness: 0.0,
    };
    c2.functions
        .extend_from_slice(&p1.functions[crossover_point..]);

    return (c1, c2);
}

pub fn reassortment(
    p1: &IteratedFunctionSystem,
    p2: &IteratedFunctionSystem,
) -> (IteratedFunctionSystem, IteratedFunctionSystem) {
    let mut rng = thread_rng();

    let mut all_functions = Vec::new();
    all_functions.append(&mut p1.functions.clone());
    all_functions.append(&mut p2.functions.clone());

    let mut shuffled_functions = all_functions.clone();
    shuffled_functions.shuffle(&mut rng);

    let mut c1 = IteratedFunctionSystem {
        functions: Vec::new(),
        fitness: 0.0,
    };

    let mut c2 = IteratedFunctionSystem {
        functions: Vec::new(),
        fitness: 0.0,
    };

    while let Some(function) = shuffled_functions.pop() {
        c1.add_function(function);

        if let Some(function) = shuffled_functions.pop() {
            c2.add_function(function);
        }
    }

    return (c1, c2);
}

pub fn random_or_binary_mutation(ifs: &mut IteratedFunctionSystem) {
    let mut rng = thread_rng();

    let mutated_functions: Vec<[f64; 6]> = ifs
        .functions
        .iter()
        .map(|func| {
            func.iter()
                .map(|coefficient| {
                    if rng.gen::<f64>() < PARAMS.mutation_probability {
                        if rng.gen::<f64>() < 0.5 {
                            mutate_random(*coefficient)
                        } else {
                            mutate_binary(*coefficient)
                        }
                    } else {
                        *coefficient
                    }
                })
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap()
        })
        .collect();
}

fn mutate_random(coefficient: f64) -> f64 {
    let mut rng = thread_rng();

    let a = PARAMS.min_singel_coefficient;
    let b = PARAMS.max_singel_coefficient;

    rng.sample(Uniform::new(a, b))
}

fn mutate_binary(coefficient: f64) -> f64 {
    let mut rng = thread_rng();

    let mut bits = coefficient.to_bits();

    for i in 0..64 {
        if rng.gen::<f64>() < 0.5 {
            bits ^= 1 << i;
        }
    }

    f64::from_bits(bits)
}

pub fn gaussian_mutation(ifs: &mut IteratedFunctionSystem) {
    let mut rng = rand::thread_rng();

    let a = PARAMS.min_singel_coefficient;
    let b = PARAMS.max_singel_coefficient;
    let r = PARAMS.gaussian_mutation_radius;

    let mutated_functions: Vec<[f64; 6]> = ifs
        .functions
        .iter()
        .map(|func| {
            func.iter()
                .map(|coefficient| {
                    if rng.gen::<f64>() < PARAMS.mutation_probability {
                        rng.sample(Normal::new(*coefficient, r * (b - a)).unwrap())
                            .max(a)
                            .min(b)
                    } else {
                        *coefficient
                    }
                })
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap()
        })
        .collect();

    ifs.functions = mutated_functions;
}
