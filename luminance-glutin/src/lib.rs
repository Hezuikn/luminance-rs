//! The [glutin](https://crates.io/crates/glutin) platform crate for [luminance](https://crates.io/crates/luminance).

#![deny(missing_docs)]

use glutin::{
  context::PossiblyCurrentContext,
  surface::{SurfaceTypeTrait, GlSurface},
};
use luminance::context::GraphicsContext;
use luminance::framebuffer::{Framebuffer, FramebufferError};
use luminance::texture::Dim2;
pub use luminance_gl::gl33::StateQueryError;
use luminance_gl::GL33;

/// The Glutin surface.
///
/// You want to create such an object in order to use any [luminance] construct.
///
/// [luminance]: https://crates.io/crates/luminance
pub struct GlutinSurface<T: SurfaceTypeTrait> {
  /// The context.
  pub ctx: PossiblyCurrentContext,
  /// The surface.
  pub surface: glutin::surface::Surface<T>,
  /// Underlying size (in physical pixels) of the surface.
  pub size: [u32; 2],
  /// OpenGL 3.3 state.
  pub gl: GL33,
}

unsafe impl<T: SurfaceTypeTrait> GraphicsContext for GlutinSurface<T> {
  type Backend = GL33;

  fn backend(&mut self) -> &mut Self::Backend {
    &mut self.gl
  }
}

impl<T: SurfaceTypeTrait> GlutinSurface<T> {
  /// Get the underlying size (in physical pixels) of the surface.
  ///
  /// This is equivalent to getting the inner size of the windowed context and converting it to
  /// a physical size by using the HiDPI factor of the windowed context.
  pub fn size(&self) -> [u32; 2] {
    self.size
  }

  /// Get access to the back buffer.
  pub fn back_buffer(&mut self) -> Result<Framebuffer<GL33, Dim2, (), ()>, FramebufferError> {
    Framebuffer::back_buffer(self, self.size())
  }

  /// Swap the back and front buffers.
  pub fn swap_buffers(&self) -> glutin::error::Result<()> {
    self.surface.swap_buffers(&self.ctx)
  }
}
