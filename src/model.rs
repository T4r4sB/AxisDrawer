use crate::points3d::*;

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    position: Point,
    color: u32,
}

#[derive(Debug, Clone)]
pub struct Model {
    vertices: Vec<Vertex>,
    indices: Vec<usize>,
}

impl Model {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn new_cylinder(a: Point, b: Point, r: f32, color: u32) -> Self {
        let delta = b - a;
        let n1 = delta.any_perp().norm();
        let n2 = cross(delta, n1).norm();
        let edge_count = 64;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for i in 0..edge_count {
            let angle = i as f32 / edge_count as f32 * std::f32::consts::PI * 2.0;
            let (s, c) = angle.sin_cos();
            let r = (n1.scale(c) + n2.scale(s)).scale(r);
            vertices.push(Vertex {
                position: a + r,
                color,
            });
            vertices.push(Vertex {
                position: b + r,
                color,
            });
            indices.push(i * 2);
            indices.push(i * 2 + 1);
            indices.push((i * 2 + 2) & (edge_count * 2 - 1));
            indices.push(i * 2 + 1);
            indices.push(i * 2);
            indices.push((i * 2 + edge_count * 2 - 1) & (edge_count * 2 - 1));
        }

        Self { vertices, indices }
    }

    pub fn to_flat_arrays(&self) -> (Vec<f32>, Vec<u16>) {
      let mut res_v = Vec::<f32>::new();
      let mut res_i = Vec::<u16>::new();
      for v in &self.vertices {
        res_v.push(v.position.x);
        res_v.push(v.position.y);
        res_v.push(v.position.z);
        res_v.push((v.color >> 16 & 0xff) as f32 / 255.0);
        res_v.push((v.color >> 8 & 0xff) as f32 / 255.0);
        res_v.push((v.color & 0xff) as f32 / 255.0);
      }
      for i in &self.indices {
        res_i.push(*i as u16);
      }

      (res_v, res_i)
    }

    pub fn append(&mut self,  other: Self) {
      for i in other.indices {
        self.indices.push(i + self.vertices.len());
      }
      for v in other.vertices {
        self.vertices.push(v);
      }
    }
}
