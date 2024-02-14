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
    let scale_cross_point = rand::thread_rng().gen_range(0..4);
    let trans_cross_point = rand::thread_rng().gen_range(4..6);

    let mut c1 = IteratedFunctionSystem {
        functions: Vec::new(),
        fitness: 0.0,
    };

    let mut c2 = IteratedFunctionSystem {
        functions: Vec::new(),
        fitness: 0.0,
    };

    let mut fs1_new = [0.0; 6];
    let mut fs2_new = [0.0; 6];

    for (f1, f2) in p1.functions.iter().zip(p2.functions.iter()) {
        if rand::thread_rng().gen::<f64>() < 0.5 {
            let scale_1: [f64; 4] = f1[..scale_cross_point]
                .iter()
                .copied()
                .chain(f2[scale_cross_point..4].iter().copied())
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();

            let trans_1: [f64; 2] = f1[4..trans_cross_point]
                .iter()
                .copied()
                .chain(f2[trans_cross_point..].iter().copied())
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();

            let scale_2: [f64; 4] = f2[..scale_cross_point]
                .iter()
                .copied()
                .chain(f1[scale_cross_point..4].iter().copied())
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();

            let trans_2: [f64; 2] = f2[4..trans_cross_point]
                .iter()
                .copied()
                .chain(f1[trans_cross_point..].iter().copied())
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap();

            fs1_new[..4].copy_from_slice(&scale_1);
            fs1_new[4..].copy_from_slice(&trans_1);

            fs2_new[..4].copy_from_slice(&scale_2);
            fs2_new[4..].copy_from_slice(&trans_2);

            c1.functions.push(fs1_new);
            c2.functions.push(fs2_new);
        } else {
            c1.functions.push(*f1);
            c2.functions.push(*f2);
        }
    }

    (c1, c2)
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

    (c1, c2)
}

pub fn random_or_binary_mutation(ifs: &mut IteratedFunctionSystem) {
    let mut rng = thread_rng();

    let _: Vec<[f64; 6]> = ifs
        .functions
        .iter_mut()
        .map(|func| {
            func.iter_mut()
                .enumerate()
                .map(|(i, coefficient)| {
                    if rng.gen::<f64>() < PARAMS.mutation_probability {
                        if rng.gen::<f64>() < 0.5 {
                            mutate_random(i)
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

fn mutate_random(i: usize) -> f64 {
    let mut rng = thread_rng();

    let (a, b) = match i {
        0..=3 => (-1.0, 1.0),
        _ => (PARAMS.min_singel_coefficient, PARAMS.max_singel_coefficient),
    };
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

    let r = PARAMS.gaussian_mutation_radius;

    let mutated_functions: Vec<[f64; 6]> = ifs
        .functions
        .iter()
        .map(|func| {
            func.iter()
                .enumerate()
                .map(|(i, coefficient)| {
                    if rng.gen::<f64>() < PARAMS.mutation_probability {
                        let (a, b) = match i {
                            0..=3 => (-1.0, 1.0),
                            _ => (PARAMS.min_singel_coefficient, PARAMS.max_singel_coefficient),
                        };
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
