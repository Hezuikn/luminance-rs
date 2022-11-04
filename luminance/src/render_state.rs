//! GPU render state.
//!
//! Such a state controls how the GPU must operate some fixed pipeline functionality, such as the
//! blending, depth test or face culling operations.

use crate::{
  blending::{Blending, BlendingMode},
  depth_stencil::{Comparison, StencilOperations, StencilTest, Write},
  face_culling::FaceCulling,
  scissor::ScissorRegion,
};

/// GPU render state.
///
/// You can get a default value with `RenderState::default` and set the operations you want with the
/// various `RenderState::set_*` methods.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderState {
  /// Blending configuration.
  pub blending: Option<BlendingMode>,
  /// Depth test configuration.
  pub depth_test: Option<Comparison>,
  /// Depth write configuration.
  pub depth_write: Write,
  /// Stencil test configuration.
  pub stencil_test: Option<StencilTest>,
  /// Stencil operations.
  pub stencil_operations: StencilOperations,
  /// Face culling configuration.
  pub face_culling: Option<FaceCulling>,
  /// Scissor region configuration.
  pub scissor: Option<ScissorRegion>,
}

impl RenderState {
  /// Override the blending configuration.
  pub fn set_blending<B>(self, blending: B) -> Self
  where
    B: Into<Option<Blending>>,
  {
    RenderState {
      blending: blending.into().map(|x| x.into()),
      ..self
    }
  }

  /// Override the blending configuration using separate blending.
  pub fn set_blending_separate(self, blending_rgb: Blending, blending_alpha: Blending) -> Self {
    RenderState {
      blending: Some(BlendingMode::Separate {
        rgb: blending_rgb,
        alpha: blending_alpha,
      }),
      ..self
    }
  }

  /// Blending configuration.
  pub fn blending(&self) -> Option<BlendingMode> {
    self.blending
  }

  /// Override the depth test configuration.
  pub fn set_depth_test<D>(self, depth_test: D) -> Self
  where
    D: Into<Option<Comparison>>,
  {
    let depth_test = depth_test.into();
    RenderState { depth_test, ..self }
  }

  /// Depth test configuration.
  pub fn depth_test(&self) -> Option<Comparison> {
    self.depth_test
  }

  /// Override the depth write configuration.
  pub fn set_depth_write(self, depth_write: Write) -> Self {
    RenderState {
      depth_write,
      ..self
    }
  }

  /// Depth write configuration.
  pub fn depth_write(&self) -> Write {
    self.depth_write
  }

  /// Override the stencil test configuration.
  pub fn set_stencil_test(self, stencil_test: impl Into<Option<StencilTest>>) -> Self {
    let stencil_test = stencil_test.into();

    RenderState {
      stencil_test,
      ..self
    }
  }

  /// Stencil test configuration.
  pub fn stencil_test(&self) -> Option<&StencilTest> {
    self.stencil_test.as_ref()
  }

  /// Override the stencil operations.
  pub fn set_stencil_operations(self, stencil_operations: StencilOperations) -> Self {
    RenderState {
      stencil_operations,
      ..self
    }
  }

  /// Stencil test operations.
  pub fn stencil_operations(&self) -> &StencilOperations {
    &self.stencil_operations
  }

  /// Override the face culling configuration.
  pub fn set_face_culling<FC>(self, face_culling: FC) -> Self
  where
    FC: Into<Option<FaceCulling>>,
  {
    RenderState {
      face_culling: face_culling.into(),
      ..self
    }
  }

  /// Face culling configuration.
  pub fn face_culling(&self) -> Option<FaceCulling> {
    self.face_culling
  }

  /// Override the scissor configuration.
  pub fn set_scissor<SR>(self, scissor: SR) -> Self
  where
    SR: Into<Option<ScissorRegion>>,
  {
    RenderState {
      scissor: scissor.into(),
      ..self
    }
  }

  /// Get the scissor configuration.
  pub fn scissor(&self) -> &Option<ScissorRegion> {
    &self.scissor
  }
}

impl Default for RenderState {
  /// The default `RenderState`.
  ///
  ///   - `blending`: `None`
  ///   - `depth_test`: `Some(Comparison::Less)`
  ///   - `depth_write`: `Write::On`
  ///   - `stencil_test`: `None`
  ///   - `stencil_operations`: `StencilOperations::default()`
  ///   - `face_culling`: `None`
  ///   - 'scissor_region`: `None`
  fn default() -> Self {
    RenderState {
      blending: None,
      depth_test: Some(Comparison::Less),
      depth_write: Write::On,
      stencil_test: None,
      stencil_operations: StencilOperations::default(),
      face_culling: None,
      scissor: None,
    }
  }
}
