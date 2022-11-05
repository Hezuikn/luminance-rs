//! OpenGL 3.3 backend.
//!
//! This module implements an OpenGL 3.3 backend for luminance. The backend type is [`GL33`].

mod buffer;
mod depth_stencil;
mod framebuffer;
mod pipeline;
mod pixel;
mod query;
pub mod shader;
mod state;
mod tess;
pub mod texture;
mod vertex_restart;

pub use self::state::GLState;
pub use self::state::StateQueryError;
use std::cell::RefCell;
use std::rc::Rc;

/// An OpenGL 3.3 backend.
///
/// This type is to be used as a luminance backend type. It implements the whole public API.
#[derive(Debug)]
pub struct GL33 {
  pub state: Rc<RefCell<GLState>>,
}

impl GL33 {
  /// Create a new OpenGL 3.3 backend.
  pub fn new() -> Result<Self, StateQueryError> {
    GLState::new().map(|state| GL33 {
      state: Rc::new(RefCell::new(state)),
    })
  }

  /// Internal access to the backend state.
  ///
  /// # Unsafety
  ///
  /// This method is **highly unsafe** as it exposes the internals of the backend. Playing with it should be done with
  /// extreme caution.
  pub unsafe fn state(&self) -> &Rc<RefCell<GLState>> {
    &self.state
  }
}
