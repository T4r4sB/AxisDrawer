use crate::model::*;
use crate::points3d::*;
use rand::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Deserialize, Serialize)]
pub struct Edge {
    v1_index: usize,
    v2_index: usize,
    length_index: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AxisSystem {
    #[serde(default = "default_distortion")]
    max_distortion: f32,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Distortion {
    bad_cnt: usize,
    max_ratio: f32,
}

impl Distortion {
    pub fn new() -> Self {
        Self {
            bad_cnt: usize::MAX,
            max_ratio: f32::MAX,
        }
    }
}

pub struct Solver {
    rng: RefCell<rngs::ThreadRng>,
    axis_system: AxisSystem,
    no_edges: Vec<(usize, usize)>,
    verteces: Vec<Point>,
    lengths_count: usize,
    prev_distortion: Distortion,
    solved: bool,
    splitting: bool,
    prev_move_result: f32,
}

static MIN_DIST: f32 = 0.01;
static COLORS: [u32; 8] = [
    0xff0000, 0x0000ff, 0x00ff00, 0xffc000, 0xff00ff, 0x804000, 0x80c0ff, 0x80ff00,
];
static WIDTH: f32 = 0.02;
fn default_distortion() -> f32 {
    1.001
}

impl Solver {
    pub fn new(axis_system: AxisSystem) -> Self {
        let mut rng = rand::thread_rng();
        let mut verteces = Vec::new();
        let mut fixed_axis_system = AxisSystem {
            edges: Vec::new(),
            max_distortion: axis_system.max_distortion,
        };
        let mut used_edges = HashSet::new();
        let mut no_edges = Vec::new();

        let mut v_id_to_index = HashMap::new();
        let mut v_index_to_id = Vec::new();

        let mut l_id_to_index = HashMap::new();
        let mut l_index_to_id = Vec::new();

        for edge in &axis_system.edges {
            if edge.v1_index == edge.v2_index {
                panic!("Egde {}:{} has same indices!", edge.v1_index, edge.v2_index);
            }

            let mut vi_by_id = |v_id| {
                if let Some(&new_index) = v_id_to_index.get(&v_id) {
                    new_index
                } else {
                    let new_index = v_index_to_id.len();
                    v_id_to_index.insert(v_id, new_index);
                    v_index_to_id.push(v_id);
                    verteces.push(Self::random_with(&mut rng));
                    new_index
                }
            };

            let mut vi = [vi_by_id(edge.v1_index), vi_by_id(edge.v2_index)];
            vi.sort();

            let li = if let Some(&li) = l_id_to_index.get(&edge.length_index) {
                li
            } else {
                let new_index = l_index_to_id.len();
                l_id_to_index.insert(edge.length_index, new_index);
                l_index_to_id.push(new_index);
                new_index
            };

            if !used_edges.insert((vi[0], vi[1])) {
                panic!("Edge {}:{} used twice!", edge.v1_index, edge.v2_index);
            }

            fixed_axis_system.edges.push(Edge {
                v1_index: vi[0],
                v2_index: vi[1],
                length_index: li,
            });
        }

        for vi0 in 0..verteces.len() {
            for vi1 in vi0 + 1..verteces.len() {
                if !used_edges.contains(&(vi0, vi1)) {
                    no_edges.push((vi0, vi1));
                }
            }
        }

        let lengths_count = l_index_to_id.len();

        Self {
            rng: RefCell::new(rng),
            axis_system: fixed_axis_system,
            no_edges,
            verteces,
            lengths_count,
            prev_distortion: Distortion::new(),
            solved: false,
            splitting: true,
            prev_move_result: f32::MAX,
        }
    }

    pub fn solved(&self) -> bool {
        self.solved
    }

    fn random_with(rng: &mut rngs::ThreadRng) -> Point {
        let h: f32 = rng.gen_range(-1.0..1.0);
        let r = (1.0 - h * h).sqrt();
        let a: f32 = rng.gen_range(0.0..std::f32::consts::PI * 2.0);
        let (s, c) = a.sin_cos();
        Point {
            x: r * c,
            y: r * s,
            z: h,
        }
    }

    fn random(&self) -> Point {
        Self::random_with(&mut self.rng.borrow_mut())
    }

    fn normalize_or_random(&self, p: Point) -> Point {
        let l = p.sqr_len();
        if l < MIN_DIST {
            self.random()
        } else {
            p.scale(1.0 / l.sqrt())
        }
    }

    pub fn distortion(&self) -> Distortion {
        let mut lengths = vec![(f32::MIN, f32::MAX); self.lengths_count];
        for edge in &self.axis_system.edges {
            let v1 = self.verteces[edge.v1_index];
            let v2 = self.verteces[edge.v2_index];
            let len = (v1 - v2).len();
            let l = &mut lengths[edge.length_index];
            l.0 = f32::max(l.0, len);
            l.1 = f32::min(l.1, len);
        }

        let mut max_length = 0.0;
        let mut max_ratio = 0.0;
        for l in &lengths {
            max_length = f32::max(max_length, l.0);
            max_ratio = f32::max(max_ratio, f32::max(l.0, MIN_DIST) / l.1);
        }

        let mut bad_cnt = 0;
        for (v1_index, v2_index) in &self.no_edges {
            let v1 = self.verteces[*v1_index];
            let v2 = self.verteces[*v2_index];
            let len = (v1 - v2).len();
            if len < max_length {
                bad_cnt += 1;
            }
            max_ratio = f32::max(max_ratio, max_length / len);
        }

        Distortion { bad_cnt, max_ratio }
    }

    pub fn adjust_dist(&mut self, i: usize, j: usize, l: f32, only_if_less: bool, factor: f32) {
        let pi = self.verteces[i];
        let pj = self.verteces[j];
        let delta = (l - (pi - pj).len()) * factor;
        if delta > 0.0 || only_if_less {
            let v = self.normalize_or_random(pi - pj).scale(delta);
            self.verteces[i] = pi + v;
            self.verteces[j] = pj - v;
        }
    }

    pub fn normalize_and_compare(&mut self, old_verteces: Vec<Point>) -> f32 {
        let mut result = 0.0;
        #[allow(clippy::needless_range_loop)]
        for i in 0..old_verteces.len() {
            self.verteces[i] = self.normalize_or_random(self.verteces[i]);
            result = f32::max(result, (old_verteces[i] - self.verteces[i]).len());
        }
        result
    }

    pub fn split_move(&mut self) -> f32 {
        let old_verteces = self.verteces.clone();
        for i in 0..self.verteces.len() {
            for j in i + 1..self.verteces.len() {
                self.adjust_dist(i, j, 2.0, false, 0.05);
            }
        }
        self.normalize_and_compare(old_verteces)
    }

    pub fn physical_move(&mut self) -> f32 {
        let mut lengths = vec![(0.0, 0.0); self.lengths_count];
        let mut deltas = vec![Point::zero(); self.verteces.len()];
        let old_verteces = self.verteces.clone();
        let mut max_l = f32::MIN;

        for edge in &self.axis_system.edges {
            let new_l = (self.verteces[edge.v1_index] - self.verteces[edge.v2_index]).len();
            lengths[edge.length_index].0 += new_l;
            lengths[edge.length_index].1 += 1.0;
            max_l = f32::max(max_l, new_l);
        }

        for edge in &self.axis_system.edges {
            let p1 = self.verteces[edge.v1_index];
            let p2 = self.verteces[edge.v2_index];
            let l = lengths[edge.length_index].0 / lengths[edge.length_index].1;
            let delta = (l - (p1 - p2).len()) * 0.01;
            let v = self.normalize_or_random(p1 - p2).scale(delta);
            deltas[edge.v1_index] += v;
            deltas[edge.v2_index] -= v;
        }

        for &(i1, i2) in &self.no_edges {
            let p1 = self.verteces[i1];
            let p2 = self.verteces[i2];
            let l = max_l;
            let delta = (l - (p1 - p2).len()) * 0.01;
            if delta > 0.0 {
                let v = self.normalize_or_random(p1 - p2).scale(delta);
                deltas[i1] += v;
                deltas[i2] -= v;
            }
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..self.verteces.len() {
            self.verteces[i] += deltas[i];
        }

        self.normalize_and_compare(old_verteces)
    }

    pub fn step(&mut self) -> Distortion {
        if self.solved {
            return self.prev_distortion;
        }

        self.prev_distortion = self.distortion();
        if self.splitting {
            let split_result = self.split_move();
            if split_result < 1.0e-3 || split_result > self.prev_move_result {
                self.splitting = false;
                self.prev_move_result = f32::MAX;
            } else {
                self.prev_move_result = split_result;
            }
        } else {
            let physical_result = self.physical_move();
            if physical_result < 1.0 || physical_result > self.prev_move_result {
                if self.prev_distortion.bad_cnt > 0 {
                    let i = self.rng.borrow_mut().gen_range(0..self.verteces.len());
                    let j = self.rng.borrow_mut().gen_range(0..self.verteces.len());

                    self.verteces.swap(i, j);
                    let new_distortion = self.distortion();
                    if new_distortion > self.prev_distortion {
                        // restore
                        self.verteces.swap(i, j);
                    } else {
                        self.prev_distortion = new_distortion;
                    }
                } else if self.prev_distortion.max_ratio < self.axis_system.max_distortion
                    && physical_result < 1.0e-5
                {
                    self.solved = true;
                }
            }
            self.prev_move_result = physical_result;
        }

        println!("distortion={:?}", self.prev_distortion);
        self.prev_distortion
    }

    pub fn generate_model(&self) -> Model {
        let mut result = Model::new();
        for v in &self.verteces {
            let vm = Model::new_cylinder(v.scale(0.8), v.scale(1.2), WIDTH * 2.0, 0);
            result.append(vm);
        }

        for edge in &self.axis_system.edges {
            let v1 = self.verteces[edge.v1_index];
            let v2 = self.verteces[edge.v2_index];
            let color = COLORS[edge.length_index & (COLORS.len() - 1)];
            let em = Model::new_cylinder(v1, v2, WIDTH, color);
            result.append(em);
        }

        result
    }
}
