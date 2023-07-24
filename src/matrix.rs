use crate::points3d::*;

#[derive(Debug, Clone, Copy)]
pub struct Matrix {
    data: [[f32; 4]; 4],
}

impl Matrix {
    pub fn new() -> Self {
        Self {
            data: [[0.0; 4]; 4],
        }
    }

    pub fn new_proj(fovy: f32, aspect: f32, zfar: f32, znear: f32) -> Self {
        let ctg = (fovy * 0.5).tan().recip();

        Self {
            data: [
                [ctg / aspect, 0.0, 0.0, 0.0],
                [0.0, ctg, 0.0, 0.0],
                [0.0, 0.0, (zfar + znear) / (zfar - znear), 1.0],
                [0.0, 0.0, (-2.0 * zfar * znear) / (zfar - znear), 0.0],
            ],
        }
    }

    pub fn translated(pos: Point) -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [pos.x, pos.y, pos.z, 1.0],
            ],
        }
    }

    pub fn rotated_x(angle_x: f32) -> Self {
        let (sx, cx) = angle_x.sin_cos();
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, cx, sx, 0.0],
                [0.0, -sx, cx, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn rotated_y(angle_y: f32) -> Self {
        let (sy, cy) = angle_y.sin_cos();
        Self {
            data: [
                [cy, 0.0, sy, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-sy, 0.0, cy, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn new_view(pos: Point, angle_y: f32,angle_x: f32) -> Self {
        mat_mul(Self::translated(pos), mat_mul(Self::rotated_y(angle_y), Self::rotated_x(angle_x)))
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.data[0].as_ptr()
    }
}

pub fn mat_mul(left: Matrix, right: Matrix) -> Matrix {
    let mut result = Matrix::new();
    for j in 0..4 {
        for i in 0..4 {
            for k in 0..4 {
                result.data[j][i] += left.data[j][k] * right.data[k][i];
            }
        }
    }
    result
}
