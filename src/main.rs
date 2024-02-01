mod ifs;
use ifs::{Barnsley, Sierpinski};

// ifs.add_function(np.array([0, 0, 0, 0.16, 0, 0.0]))
// ifs.add_function(np.array([0.85, 0.04, -0.04, 0.85, 0.0, 1.6]))
// ifs.add_function(np.array([0.2, -0.26, 0.23, 0.22, 0.0, 1.6]))
// ifs.add_function(np.array([-0.15, 0.28, 0.26, 0.24, 0.0, 0.44]))
fn main() {
    // let funcs: Vec<[f64; 6]> = vec![
    //     [0., 0., 0., 0.16, 0., 0.],
    //     [0., 0., 0., 0.16, 0., 0.0],
    //     [0.85, 0.04, -0.04, 0.85, 0.0, 1.6],
    //     [0.2, -0.26, 0.23, 0.22, 0.0, 1.6],
    //     [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44],
    // ];

    // let mut ifs = ifs::IteratedFunctionSystem {
    //     functions: funcs,
    //     fitness: 0.0,
    // };
    let mut ifs = ifs::IteratedFunctionSystem::barnsley_fern();

    println!("{:?}", ifs);

    let point = (1.0, -1.0);

    println!("{:?}", point);
    let p2 = ifs.apply_function(point);
    println!("{:?}", p2);

    let pts = ifs.generate_points(10, (0.0, 0.0));
    println!("{:?}", pts);

    //ifs.plot_fractal(10_000_000, (0.0, 0.0));
}
