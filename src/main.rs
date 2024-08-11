// Modification from https://rajeshrinet.github.io/blog/2014/ising-model/

use ndarray::Array2;
use rand::distributions::{Distribution, Uniform};

type LatticeItem = i8;

fn main() {
    let n = 100;
    let eq_steps = 1024;
    let mc_steps = 1024;

    let from_tmp = 1.53;
    let to_tmp = 3.28;

    let mut t = from_tmp;

    let mut state = init_array(n);

    while t < to_tmp {
        let i_t = 1.0 / t;

        for _ in 0..eq_steps {
            monte_carlo_move(&mut state, i_t);
        }

        // for _ in 0..mc_steps {
        //     monte_carlo_move(&mut state, i_t);
        // }

        let energy = calculate_energy(&state) / n as f64 / n as f64;
        let magnetization = calculate_magnetization(&state) / n as f64 / n as f64;

        println!("temperature = {t:.2}, energy = {energy:.2}, magnetization={magnetization:.2}");

        t += (to_tmp - from_tmp) / 50.0;
    }
}

fn init_array(n: usize) -> Array2<LatticeItem> {
    let mut array = ndarray::Array2::zeros((n, n));
    for elem in &mut array {
        if rand::random() {
            *elem = 1;
        } else {
            *elem = -1
        }
    }

    array
}

fn monte_carlo_move(state: &mut Array2<LatticeItem>, beta: f64) {
    let &[x, y] = state.shape() else { unreachable!() };

    let x_rand = Uniform::new(0, x);
    let y_rand = Uniform::new(0, y);
    let uniform = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();


    for _ in 0..x {
        for _ in 0..y {
            let a = x_rand.sample(&mut rng);
            let b = y_rand.sample(&mut rng);
            let s = state[[a, b]];
            let nb = state[[(a + 1) % x, b]] as i64 + state[[a, (b + 1) % y]] as i64 + state[[(a + x - 1) % x, b]] as i64 + state[[a, (b + y - 1) % y]] as i64;
            let cost = 2 * s as i64 * nb;
            if cost < 0 {
                state[[a, b]] *= -1;
            } else if uniform.sample(&mut rng) < f64::exp(-cost as f64 * beta) {
                state[[a, b]] *= -1;
            }
        }
    }
}

fn calculate_energy(state: &Array2<LatticeItem>) -> f64 {
    let mut energy = 0.0;
    let &[x, y] = state.shape() else { unreachable!() };
    for a in 0..x {
        for b in 0..y {
            let s = state[[a, b]];
            let nb = state[[(a + 1) % x, b]] as i64 + state[[a, (b + 1) % y]] as i64 + state[[(a - 1) % x, b]] as i64 + state[[a, (b - 1) % y]] as i64;
            energy += (-nb * s as i64) as f64;
        }
    }

    energy / 4.0
}

fn calculate_magnetization(state: &Array2<LatticeItem>) -> f64 {
    state.iter().map(|i| *i as f64).sum()
}