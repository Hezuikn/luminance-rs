//! The [glutin](https://crates.io/crates/glutin) platform crate for [luminance](https://crates.io/crates/luminance).

#![deny(missing_docs)]

use gl; //todo does this belong?
use glutin::{
  event_loop::EventLoop, window::WindowBuilder, Api, ContextBuilder, ContextError, CreationError,
  GlProfile, GlRequest, NotCurrent, PossiblyCurrent, WindowedContext,
};
use luminance::context::Context;
use luminance_gl2::GL33;
use std::error;
use std::fmt;
use std::os::raw::c_void;

/// Error that might occur when creating a Glutin surface.
#[derive(Debug)]
pub enum GlutinSurfaceError {
  /// Something went wrong when creating the Glutin surface. The carried [`CreationError`] provides
  /// more information.
  CreationError(CreationError),
  /// OpenGL context error.
  ContextError(ContextError),
  /// Error with the backend.
  BackendError(String),
}

impl fmt::Display for GlutinSurfaceError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      GlutinSurfaceError::CreationError(ref e) => write!(f, "Glutin surface creation error: {}", e),
      GlutinSurfaceError::ContextError(ref e) => {
        write!(f, "Glutin OpenGL context creation error: {}", e)
      }
      GlutinSurfaceError::BackendError(ref reason) => {
        write!(f, "Glutin surface backend error: {}", reason)
      }
    }
  }
}

impl error::Error for GlutinSurfaceError {
  fn source(&self) -> Option<&(dyn error::Error + 'static)> {
    match self {
      GlutinSurfaceError::CreationError(e) => Some(e),
      GlutinSurfaceError::ContextError(e) => Some(e),
      GlutinSurfaceError::BackendError(_) => None,
    }
  }
}

impl From<CreationError> for GlutinSurfaceError {
  fn from(e: CreationError) -> Self {
    GlutinSurfaceError::CreationError(e)
  }
}

impl From<ContextError> for GlutinSurfaceError {
  fn from(e: ContextError) -> Self {
    GlutinSurfaceError::ContextError(e)
  }
}

/// The Glutin surface.
///
/// You want to create such an object in order to use any [luminance] construct.
///
/// [luminance]: https://crates.io/crates/luminance
pub struct GlutinSurface {
  /// The windowed context.
  pub window_ctx: WindowedContext<PossiblyCurrent>,

  /// Wrapped luminance context.
  pub ctx: Context<GL33>,
}

impl GlutinSurface {
  /// Create a new [`GlutinSurface`] by consuming a [`WindowBuilder`].
  ///
  /// This is an alternative method to [`new_gl33`] that is more flexible as you have access to the
  /// whole `glutin` types.
  ///
  /// `window_builder` is the default object when passed to your closure and `ctx_builder` is
  /// already initialized for the OpenGL context (you’re not supposed to change it!).
  ///
  /// [`new_gl33`]: crate::GlutinSurface::new_gl33
  pub fn new_gl33_from_builders<'a, WB, CB>(
    window_builder: WB,
    ctx_builder: CB,
  ) -> Result<(Self, EventLoop<()>), GlutinSurfaceError>
  where
    WB: FnOnce(&mut EventLoop<()>) -> WindowBuilder,
    CB:
      FnOnce(&mut EventLoop<()>, ContextBuilder<'a, NotCurrent>) -> ContextBuilder<'a, NotCurrent>,
  {
    let mut event_loop = EventLoop::new();

    let window_builder = window_builder(&mut event_loop);

    let windowed_ctx = ctx_builder(
      &mut event_loop,
      ContextBuilder::new()
        .with_gl(GlRequest::Specific(Api::OpenGl, (3, 3)))
        .with_gl_profile(GlProfile::Core),
    )
    .build_windowed(window_builder, &event_loop)?;

    let window_ctx = unsafe { windowed_ctx.make_current().map_err(|(_, e)| e)? };

    // init OpenGL
    gl::load_with(|s| window_ctx.get_proc_address(s) as *const c_void);

    window_ctx.window().set_visible(true);

    let ctx = Context::new(GL33::new)
      .ok_or_else(|| GlutinSurfaceError::BackendError("unavailable OpenGL 3.3 state".to_owned()))?;
    let surface = GlutinSurface { ctx, window_ctx };

    Ok((surface, event_loop))
  }

  /// Create a new [`GlutinSurface`] from scratch.
  pub fn new_gl33(
    window_builder: WindowBuilder,
    samples: u16,
  ) -> Result<(Self, EventLoop<()>), GlutinSurfaceError> {
    Self::new_gl33_from_builders(
      |_el| window_builder,
      |_el, cb| {
        cb.with_multisampling(samples)
          .with_double_buffer(Some(true))
      },
    )
  }

  /// Get the underlying size (in physical pixels) of the surface.
  ///
  /// This is equivalent to getting the inner size of the windowed context and converting it to
  /// a physical size by using the HiDPI factor of the windowed context.
  pub fn size(&self) -> [u32; 2] {
    let size = self.window_ctx.window().inner_size();
    [size.width, size.height]
  }

  /// Swap the back and front buffers.
  pub fn swap_buffers(&mut self) {
    let _ = self.window_ctx.swap_buffers();
  }
}
