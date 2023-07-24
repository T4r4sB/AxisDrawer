use winit::event_loop::EventLoopBuilder;

use std::ffi::{CStr, CString};
use std::num::NonZeroU32;
use std::ops::Deref;

use gif::*;

use winit::event::{Event, WindowEvent};
use winit::window::WindowBuilder;

use raw_window_handle::HasRawWindowHandle;

use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::SwapInterval;

use glutin_winit::{self, DisplayBuilder, GlWindow};

mod points3d;
use crate::points3d::*;
mod matrix;
use crate::matrix::*;
mod model;
mod solver;
use crate::solver::*;
mod loader;
use crate::loader::*;

pub mod gl {
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));

    pub use Gles2 as Gl;
}

pub fn gl_main(event_loop: winit::event_loop::EventLoop<()>) {
    // Only windows requires the window to be present before creating the display.
    // Other platforms don't really need one.
    //
    // XXX if you don't care about running on android or so you can safely remove
    // this condition and always pass the window builder.
    let window_builder = if cfg!(wgl_backend) {
        Some(WindowBuilder::new().with_transparent(true))
    } else {
        None
    };

    // The template will match only the configurations supporting rendering
    // to windows.
    //
    // XXX We force transparency only on macOS, given that EGL on X11 doesn't
    // have it, but we still want to show window. The macOS situation is like
    // that, because we can query only one config at a time on it, but all
    // normal platforms will return multiple configs, so we can find the config
    // with transparency ourselves inside the `reduce`.
    let template = ConfigTemplateBuilder::new()
        .with_alpha_size(8)
        .with_transparency(cfg!(cgl_backend));

    let display_builder = DisplayBuilder::new().with_window_builder(window_builder);

    let (mut window, gl_config) = display_builder
        .build(&event_loop, template, |configs| {
            // Find the config with the maximum number of samples, so our triangle will
            // be smooth.
            configs
                .reduce(|accum, config| {
                    let transparency_check = config.supports_transparency().unwrap_or(false)
                        & !accum.supports_transparency().unwrap_or(false);

                    if transparency_check || config.num_samples() > accum.num_samples() {
                        config
                    } else {
                        accum
                    }
                })
                .unwrap()
        })
        .unwrap();

    println!("Picked a config with {} samples", gl_config.num_samples());

    let raw_window_handle = window.as_ref().map(|window| window.raw_window_handle());

    // XXX The display could be obtained from the any object created by it, so we
    // can query it from the config.
    let gl_display = gl_config.display();

    // The context creation part. It can be created before surface and that's how
    // it's expected in multithreaded + multiwindow operation mode, since you
    // can send NotCurrentContext, but not Surface.
    let context_attributes = ContextAttributesBuilder::new().build(raw_window_handle);

    // Since glutin by default tries to create OpenGL core context, which may not be
    // present we should try gles.
    let fallback_context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::Gles(None))
        .build(raw_window_handle);

    // There are also some old devices that support neither modern OpenGL nor GLES.
    // To support these we can try and create a 2.1 context.
    let legacy_context_attributes = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
        .build(raw_window_handle);

    let mut not_current_gl_context = Some(unsafe {
        gl_display
            .create_context(&gl_config, &context_attributes)
            .unwrap_or_else(|_| {
                gl_display
                    .create_context(&gl_config, &fallback_context_attributes)
                    .unwrap_or_else(|_| {
                        gl_display
                            .create_context(&gl_config, &legacy_context_attributes)
                            .expect("failed to create context")
                    })
            })
    });

    let mut state = None;
    let mut renderer = None;
    event_loop.run(move |event, window_target, control_flow| {
        // control_flow.set_wait();
        match event {
            Event::Resumed => {
                #[cfg(android_platform)]
                println!("Android window available");

                let window = window.take().unwrap_or_else(|| {
                    let window_builder = WindowBuilder::new().with_transparent(true);
                    glutin_winit::finalize_window(window_target, window_builder, &gl_config)
                        .unwrap()
                });

                let attrs = window.build_surface_attributes(<_>::default());
                let gl_surface = unsafe {
                    gl_config
                        .display()
                        .create_window_surface(&gl_config, &attrs)
                        .unwrap()
                };

                // Make it current.
                let gl_context = not_current_gl_context
                    .take()
                    .unwrap()
                    .make_current(&gl_surface)
                    .unwrap();

                // The context needs to be current for the Renderer to set up shaders and
                // buffers. It also performs function loading, which needs a current context on
                // WGL.
                renderer.get_or_insert_with(|| Renderer::new(&gl_display));

                // Try setting vsync.
                if let Err(res) = gl_surface
                    .set_swap_interval(&gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()))
                {
                    eprintln!("Error setting vsync: {res:?}");
                }

                assert!(state.replace((gl_context, gl_surface, window)).is_none());
            }
            Event::Suspended => {
                // This event is only raised on Android, where the backing NativeWindow for a GL
                // Surface can appear and disappear at any moment.
                println!("Android window removed");

                // Destroy the GL Surface and un-current the GL Context before ndk-glue releases
                // the window back to the system.
                let (gl_context, ..) = state.take().unwrap();
                assert!(not_current_gl_context
                    .replace(gl_context.make_not_current().unwrap())
                    .is_none());
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::Resized(size) => {
                    if size.width != 0 && size.height != 0 {
                        // Some platforms like EGL require resizing GL surface to update the size
                        // Notable platforms here are Wayland and macOS, other don't require it
                        // and the function is no-op, but it's wise to resize it for portability
                        // reasons.
                        if let Some((gl_context, gl_surface, _)) = &state {
                            gl_surface.resize(
                                gl_context,
                                NonZeroU32::new(size.width).unwrap(),
                                NonZeroU32::new(size.height).unwrap(),
                            );
                            let renderer = renderer.as_mut().unwrap();
                            renderer.resize(size.width as i32, size.height as i32);
                        }
                    }
                }
                WindowEvent::CloseRequested => {
                    control_flow.set_exit();
                }
                _ => (),
            },
            Event::RedrawEventsCleared => {
                if let Some((gl_context, gl_surface, window)) = &state {
                    let renderer = renderer.as_mut().unwrap();
                    renderer.draw();
                    window.request_redraw();

                    gl_surface.swap_buffers(gl_context).unwrap();
                }
            }
            _ => (),
        }
    })
}

pub struct Renderer {
    program: gl::types::GLuint,
    vao: gl::types::GLuint,
    vbo: gl::types::GLuint,
    veo: gl::types::GLuint,
    proj_matrix: Matrix,
    view_matrix: Matrix,
    solver: Solver,
    output: String,
    window_size: (usize, usize),
    gif_size: (usize, usize),
    frame_loop_length: usize,
    encoder: Option<Encoder<std::fs::File>>,
    image_buffer: Vec<u8>,
    frames: usize,
    iterations: usize,
    gl: gl::Gl,
}

impl Renderer {
    pub fn new<D: GlDisplay>(gl_display: &D) -> Self {
        unsafe {
            let gl = gl::Gl::load_with(|symbol| {
                let symbol = CString::new(symbol).unwrap();
                gl_display.get_proc_address(symbol.as_c_str()).cast()
            });

            gl.Enable(gl::DEPTH_TEST);

            if let Some(renderer) = get_gl_string(&gl, gl::RENDERER) {
                println!("Running on {}", renderer.to_string_lossy());
            }
            if let Some(version) = get_gl_string(&gl, gl::VERSION) {
                println!("OpenGL Version {}", version.to_string_lossy());
            }

            if let Some(shaders_version) = get_gl_string(&gl, gl::SHADING_LANGUAGE_VERSION) {
                println!("Shaders version on {}", shaders_version.to_string_lossy());
            }

            let vertex_shader = create_shader(&gl, gl::VERTEX_SHADER, VERTEX_SHADER_SOURCE);
            let fragment_shader = create_shader(&gl, gl::FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE);

            let program = gl.CreateProgram();

            gl.AttachShader(program, vertex_shader);
            gl.AttachShader(program, fragment_shader);

            gl.LinkProgram(program);
            gl.UseProgram(program);

            gl.DeleteShader(vertex_shader);
            gl.DeleteShader(fragment_shader);

            let mut vao = std::mem::zeroed();
            gl.GenVertexArrays(1, &mut vao);

            let mut vbo = std::mem::zeroed();
            gl.GenBuffers(1, &mut vbo);

            let mut veo = std::mem::zeroed();
            gl.GenBuffers(1, &mut veo);

            let args = parse_command_line();

            let mut path = std::path::PathBuf::from(&args.input);
            path.set_extension("json");

            let axis_system = load(&path).unwrap();
            let output = args.output.unwrap_or(args.input);
            let solver = Solver::new(axis_system);

            let frame_loop_length = args.length.unwrap_or(256);
            let frame_size = args.size.unwrap_or(256);

            Self {
                program,
                vao,
                vbo,
                veo,
                proj_matrix: Matrix::new_proj(std::f32::consts::FRAC_PI_8, 1.0, 100.0, 0.1),
                view_matrix: Matrix::new_view(
                    Point {
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                    },
                    0.0,
                    0.0,
                ),
                solver,
                output,
                window_size: (0, 0),
                gif_size: (frame_size, frame_size),
                frame_loop_length,
                encoder: None,
                image_buffer: Vec::new(),
                frames: 0,
                iterations: 0,
                gl,
            }
        }
    }

    pub fn draw(&mut self) {
        unsafe {
            self.iterations += 1;
            self.gl.UseProgram(self.program);
            let proj_matrix_location = self
                .gl
                .GetUniformLocation(self.program, b"proj\0".as_ptr() as *const _);
            self.gl.UniformMatrix4fv(
                proj_matrix_location,
                1,
                gl::FALSE,
                self.proj_matrix.as_ptr() as *const _,
            );

            let delta =
                self.iterations as f32 / self.frame_loop_length as f32 * std::f32::consts::PI * 2.0;
            let angle = delta * 2.0;
            let angle_x = delta.sin();
            let dist = 3.5;
            self.view_matrix = Matrix::new_view(
                Point {
                    x: angle.sin() * angle_x.cos() * dist,
                    y: angle_x.sin() * dist,
                    z: angle.cos() * angle_x.cos() * dist,
                },
                angle,
                angle_x,
            );

            let view_matrix_location = self
                .gl
                .GetUniformLocation(self.program, b"view\0".as_ptr() as *const _);
            self.gl.UniformMatrix4fv(
                view_matrix_location,
                1,
                gl::FALSE,
                self.view_matrix.as_ptr() as *const _,
            );

            for _ in 0..1 {
                self.solver.step();
            }

            let model = self.solver.generate_model();
            let (vertex_data, index_data) = model.to_flat_arrays();

            self.gl.BindVertexArray(self.vao);
            self.gl.BindBuffer(gl::ARRAY_BUFFER, self.vbo);
            self.gl.BufferData(
                gl::ARRAY_BUFFER,
                (vertex_data.len() * std::mem::size_of::<f32>()) as gl::types::GLsizeiptr,
                vertex_data.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            let pos_attrib = self
                .gl
                .GetAttribLocation(self.program, b"position\0".as_ptr() as *const _);
            let color_attrib = self
                .gl
                .GetAttribLocation(self.program, b"color\0".as_ptr() as *const _);
            self.gl.VertexAttribPointer(
                pos_attrib as gl::types::GLuint,
                3,
                gl::FLOAT,
                0,
                6 * std::mem::size_of::<f32>() as gl::types::GLsizei,
                std::ptr::null(),
            );
            self.gl.VertexAttribPointer(
                color_attrib as gl::types::GLuint,
                3,
                gl::FLOAT,
                0,
                6 * std::mem::size_of::<f32>() as gl::types::GLsizei,
                (3 * std::mem::size_of::<f32>()) as _,
            );

            self.gl.BindBuffer(gl::ELEMENT_ARRAY_BUFFER, self.veo);
            self.gl.BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                (index_data.len() * std::mem::size_of::<u16>()) as gl::types::GLsizeiptr,
                index_data.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            self.gl
                .EnableVertexAttribArray(pos_attrib as gl::types::GLuint);
            self.gl
                .EnableVertexAttribArray(color_attrib as gl::types::GLuint);

            self.gl.ClearColor(1.0, 1.0, 1.0, 1.0);
            self.gl.Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            self.gl.DrawElements(
                gl::TRIANGLES,
                index_data.len() as i32,
                gl::UNSIGNED_SHORT,
                std::ptr::null(),
            );

            if self.solver.solved() && self.frames < self.frame_loop_length {
                if self.frames == 0 {
                    let mut path = std::path::PathBuf::from(&self.output);
                    path.set_extension("gif");
                    let file = std::fs::File::create(&path).unwrap();
                    self.encoder = Some(
                        Encoder::new(file, self.gif_size.0 as u16, self.gif_size.1 as u16, &[])
                            .unwrap(),
                    );
                    self.encoder
                        .as_mut()
                        .unwrap()
                        .set_repeat(Repeat::Infinite)
                        .unwrap();
                }

                self.image_buffer
                    .resize(3 * self.gif_size.0 * self.gif_size.1, 0);
                self.gl.ReadPixels(
                    ((self.window_size.0 - self.gif_size.0) / 2) as i32,
                    ((self.window_size.1 - self.gif_size.1) / 2) as i32,
                    self.gif_size.0 as i32,
                    self.gif_size.1 as i32,
                    gl::RGB,
                    gl::UNSIGNED_BYTE,
                    self.image_buffer.as_mut_ptr() as *mut _,
                );

                let mut frame = gif::Frame::from_rgb_speed(
                    self.gif_size.0 as u16,
                    self.gif_size.1 as u16,
                    &self.image_buffer,
                    10, // a good compromise between speed and quality
                );
                frame.delay = (1024 / self.frame_loop_length) as u16;
                self.encoder.as_mut().unwrap().write_frame(&frame).unwrap();

                self.frames += 1;
                println!("processed {} frames", self.frames);
                if self.frames == self.frame_loop_length {
                    self.encoder = None; // finish gif processing
                    println!("saved gif");
                }
            }
        }
    }

    pub fn resize(&mut self, width: i32, height: i32) {
        unsafe {
            self.window_size.0 = width as usize;
            self.window_size.1 = height as usize;
            self.gl.Viewport(
                (self.window_size.0 as i32 - self.gif_size.0 as i32) / 2,
                (self.window_size.1 as i32 - self.gif_size.1 as i32) / 2,
                self.gif_size.0 as i32,
                self.gif_size.1 as i32,
            );
            self.proj_matrix = Matrix::new_proj(
                std::f32::consts::FRAC_PI_4,
                self.gif_size.0 as f32 / self.gif_size.1 as f32,
                100.0,
                0.1,
            );
        }
    }
}

impl Deref for Renderer {
    type Target = gl::Gl;

    fn deref(&self) -> &Self::Target {
        &self.gl
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.gl.DeleteProgram(self.program);
            self.gl.DeleteBuffers(1, &self.vbo);
            self.gl.DeleteBuffers(1, &self.veo);
        }
    }
}

unsafe fn create_shader(
    gl: &gl::Gl,
    shader: gl::types::GLenum,
    source: &[u8],
) -> gl::types::GLuint {
    let shader = gl.CreateShader(shader);
    gl.ShaderSource(
        shader,
        1,
        [source.as_ptr().cast()].as_ptr(),
        std::ptr::null(),
    );
    gl.CompileShader(shader);
    shader
}

fn get_gl_string(gl: &gl::Gl, variant: gl::types::GLenum) -> Option<&'static CStr> {
    unsafe {
        let s = gl.GetString(variant);
        (!s.is_null()).then(|| CStr::from_ptr(s.cast()))
    }
}

const VERTEX_SHADER_SOURCE: &[u8] = b"
#version 100
precision mediump float;

uniform mat4 proj;
uniform mat4 view;

attribute vec3 position;
attribute vec3 color;

varying vec3 v_color;
varying vec3 v_pos;

void main() {
    gl_Position = proj * (view * vec4(position, 1.0));
    v_color = color;
    v_pos = gl_Position.xyz;
}
\0";

const FRAGMENT_SHADER_SOURCE: &[u8] = b"
#version 100
precision mediump float;

varying vec3 v_color;
varying vec3 v_pos;

void main() {
    float factor = max(0.0, min(1.0, (4.5 - v_pos.z) * 0.5));
    vec3 white = vec3(1.0, 1.0, 1.0);
    vec3 color = white + (v_color - white) * factor;
    gl_FragColor = vec4(color, 1.0);
}
\0";

fn main() {
    gl_main(EventLoopBuilder::new().build())
}
