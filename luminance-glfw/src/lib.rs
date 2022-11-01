//! [GLFW](https://crates.io/crates/glfw) backend for [luminance](https://crates.io/crates/luminance).

#![deny(missing_docs)]

use gl;
use glfw::{self, Glfw, InitError, Window, WindowEvent};
use luminance::{
  context::GraphicsContext,
  framebuffer::{Framebuffer, FramebufferError},
  texture::Dim2,
};
pub use luminance_gl::gl33::StateQueryError;
use luminance_gl::GL33;
use std::{error, fmt, os::raw::c_void, sync::mpsc::Receiver};

/// Error that can be risen while creating a surface.
#[non_exhaustive]
#[derive(Debug)]
pub enum GlfwSurfaceError<E> {
  /// Initialization of the surface went wrong.
  ///
  /// This variant exposes a **glfw** error for further information about what went wrong.
  InitError(InitError),

  /// User error.
  UserError(E),

  /// The graphics state is not available.
  ///
  /// This error is generated when the initialization code is called on a thread on which the
  /// graphics state has already been acquired.
  GraphicsStateError(StateQueryError),
}

impl<E> fmt::Display for GlfwSurfaceError<E>
where
  E: fmt::Display,
{
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      GlfwSurfaceError::InitError(ref e) => write!(f, "initialization error: {}", e),
      GlfwSurfaceError::UserError(ref e) => write!(f, "user error: {}", e),
      GlfwSurfaceError::GraphicsStateError(ref e) => {
        write!(f, "failed to get graphics state: {}", e)
      }
    }
  }
}

impl<E> From<InitError> for GlfwSurfaceError<E> {
  fn from(e: InitError) -> Self {
    GlfwSurfaceError::InitError(e)
  }
}

impl<E> error::Error for GlfwSurfaceError<E>
where
  E: 'static + error::Error,
{
  fn source(&self) -> Option<&(dyn error::Error + 'static)> {
    match self {
      GlfwSurfaceError::InitError(e) => Some(e),
      GlfwSurfaceError::UserError(e) => Some(e),
      GlfwSurfaceError::GraphicsStateError(e) => Some(e),
    }
  }
}

/// GLFW surface.
///
/// This type is a helper that exposes two important concepts: the GLFW event receiver that you can use it with to
/// poll events and the [`GL33Context`], which allows you to perform the rendering part.
#[derive(Debug)]
pub struct GlfwSurface {
  /// Wrapped GLFW events queue.
  pub events_rx: Receiver<(f64, WindowEvent)>,

  /// Wrapped luminance context.
  pub context: GL33Context,
}

impl GlfwSurface {
  /// Initialize GLFW to provide a luminance environment.
  pub fn new<E>(
    create_window: impl FnOnce(
      &mut Glfw,
    )
      -> Result<(Window, Receiver<(f64, WindowEvent)>), GlfwSurfaceError<E>>,
  ) -> Result<Self, GlfwSurfaceError<E>> {
    let mut test = std::time::Instant::now();
    
    #[cfg(feature = "log-errors")]
    let error_cbk = glfw::LOG_ERRORS;
    #[cfg(not(feature = "log-errors"))]
    let error_cbk = glfw::FAIL_ON_ERRORS;
    
    dbg!(test.elapsed());

    let mut glfw = glfw::init(error_cbk)?;
    
    dbg!(test.elapsed());

    // OpenGL hints
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
      glfw::OpenGlProfileHint::Core,
    ));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::ContextVersionMajor(3));
    glfw.window_hint(glfw::WindowHint::ContextVersionMinor(3));
    
    dbg!(test.elapsed());

    let (mut window, events_rx) = create_window(&mut glfw)?;
    
    dbg!(test.elapsed());

    // init OpenGL
    gl::load_with(|s| window.get_proc_address(s) as *const c_void);
    
    dbg!(test.elapsed());

    let gl = GL33::new().map_err(GlfwSurfaceError::GraphicsStateError)?;
    
    dbg!(test.elapsed());
    
    let context = GL33Context { window, gl };
    let surface = GlfwSurface { events_rx, context };

    Ok(surface)
  }
}

/// Luminance OpenGL 3.3 context.
///
/// This type also re-exports the GLFW window, if you need access to it.
#[derive(Debug)]
pub struct GL33Context {
  /// Wrapped GLFW window.
  pub window: Window,

  /// OpenGL 3.3 state.
  gl: GL33,
}

impl GL33Context {
  /// Get the back buffer.
  pub fn back_buffer(&mut self) -> Result<Framebuffer<GL33, Dim2, (), ()>, FramebufferError> {
    let (w, h) = self.window.get_framebuffer_size();
    Framebuffer::back_buffer(self, [w as u32, h as u32])
  }
}

unsafe impl GraphicsContext for GL33Context {
  type Backend = GL33;

  fn backend(&mut self) -> &mut Self::Backend {
    &mut self.gl
  }
}
