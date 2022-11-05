//! Graphics state.

use crate::gl33::{
  depth_stencil::{
    comparison_to_glenum, glenum_to_comparison, glenum_to_stencil_op, stencil_op_to_glenum,
  },
  vertex_restart::VertexRestart,
};
use gl::types::*;
use luminance::{
  blending::{Equation, Factor},
  depth_stencil::{Comparison, StencilOperations, StencilTest, Write},
  face_culling::{FaceCullingMode, FaceCullingOrder},
  scissor::ScissorRegion,
};
use std::{cell::RefCell, error, ffi::CStr, fmt, marker::PhantomData, os::raw::c_char};

// TLS synchronization barrier for `GLState`.
//
// Note: disable on no_std.
thread_local!(static TLS_ACQUIRE_GFX_STATE: RefCell<Option<()>> = RefCell::new(Some(())));

#[derive(Debug)]
pub(crate) struct BindingStack {
  pub(crate) next_texture_unit: u32,
  pub(crate) free_texture_units: Vec<u32>,
  pub(crate) next_shader_data: u32,
  pub(crate) free_shader_data: Vec<u32>,
}

impl BindingStack {
  // Create a new, empty binding stack.
  fn new() -> Self {
    BindingStack {
      next_texture_unit: 0,
      free_texture_units: Vec::new(),
      next_shader_data: 0,
      free_shader_data: Vec::new(),
    }
  }
}

/// Cached value.
///
/// A cached value is used to prevent issuing costy GPU commands if we know the target value is
/// already set to what the command tries to set. For instance, if you ask to use a texture ID
/// `34` once, that value will be set on the GPU and cached on our side. Later, if no other texture
/// setting has occurred, if you ask to use the texture ID `34` again, because the value is cached,
/// we know the GPU is already using it, so we don’t have to perform anything GPU-wise.
///
/// This optimization has limits and sometimes, because of side-effects, it is not possible to cache
/// something correctly.
///
/// Note: do not confuse [`Cached`] with [`Bind`]. The latter is for internal use only and
/// is used to either use the regular cache mechanism or override it to force a value to be
/// written. It cannot be used to invalidate a setting for later use.
#[derive(Debug)]
struct Cached<T>(Option<T>)
where
  T: PartialEq;

impl<T> Cached<T>
where
  T: PartialEq,
{
  /// Cache a value.
  fn new(initial: T) -> Self {
    Cached(Some(initial))
  }

  /// Explicitly invalidate a value.
  ///
  /// This is necessary when we want to be able to force a GPU command to run.
  fn invalidate(&mut self) {
    self.0 = None;
  }

  fn set(&mut self, value: T) {
    self.0 = Some(value);
  }

  /// Check if the cached value is invalid regarding a value.
  ///
  /// A non-cached value (i.e. empty) is always invalid whatever compared value. If a value is
  /// already cached, then it’s invalid if it’s not equal ([`PartialEq`]) to the input value.
  fn is_invalid(&self, new_val: &T) -> bool {
    match &self.0 {
      Some(ref t) => t != new_val,
      _ => true,
    }
  }
}

/// The graphics state.
///
/// This type represents the current state of a given graphics context. It acts
/// as a forward-gate to all the exposed features from the low-level API but
/// adds a small cache layer over it to prevent from issuing the same API call (with
/// the same parameters).
#[derive(Debug)]
pub struct GLState {
  _a: PhantomData<*const ()>, // !Send and !Sync

  // binding stack
  binding_stack: BindingStack,

  // viewport
  viewport: Cached<[GLint; 4]>,

  // clear buffers
  clear_color: Cached<[GLfloat; 4]>,
  clear_depth: Cached<GLfloat>,
  clear_stencil: Cached<GLint>,

  // blending
  blending_state: Cached<BlendingState>,
  blending_equations: Cached<BlendingEquations>,
  blending_funcs: Cached<BlendingFactors>,

  // depth test
  depth_test: Cached<DepthTest>,
  depth_test_comparison: Cached<Comparison>,

  // depth write
  depth_write: Cached<Write>,

  // stencil test
  stencil_test_enabled: Cached<bool>,
  stencil_test: Cached<StencilTest>,
  stencil_operations: Cached<StencilOperations>,

  // face culling
  face_culling_state: Cached<FaceCullingState>,
  face_culling_order: Cached<FaceCullingOrder>,
  face_culling_mode: Cached<FaceCullingMode>,

  // scissor
  scissor_state: Cached<ScissorState>,
  scissor_region: Cached<ScissorRegion>,

  // vertex restart
  vertex_restart: Cached<VertexRestart>,

  // patch primitive vertex number
  patch_vertex_nb: Cached<usize>,

  // texture
  current_texture_unit: Cached<GLenum>,
  bound_textures: Vec<(GLenum, GLuint)>,

  // texture buffer used to optimize texture creation; regular textures typically will never ask
  // for fetching from this set but framebuffers, who often generate several textures, might use
  // this opportunity to get N textures (color, depth and stencil) at once, in a single CPU / GPU
  // roundtrip
  //
  // fishy fishy
  texture_swimming_pool: Vec<GLuint>,

  // uniform buffer
  bound_uniform_buffers: Vec<GLuint>,

  // array buffer
  bound_array_buffer: GLuint,

  // element buffer
  bound_element_array_buffer: GLuint,

  // framebuffer
  bound_draw_framebuffer: Cached<GLuint>,

  // vertex array
  bound_vertex_array: GLuint,

  // shader program
  current_program: GLuint,

  // framebuffer sRGB
  srgb_framebuffer_enabled: Cached<bool>,

  // vendor name; cached when asked the first time and then re-used
  vendor_name: Option<String>,

  // renderer name; cached when asked the first time and then re-used
  renderer_name: Option<String>,

  // OpenGL version; cached when asked the first time and then re-used
  gl_version: Option<String>,

  // GLSL version; cached when asked the first time and then re-used
  glsl_version: Option<String>,

  /// Maximum number of elements a texture array can hold.
  max_texture_array_elements: Option<usize>,
}

impl GLState {
  /// Create a new `GLState`.
  ///
  /// > Note: keep in mind you can create only one per thread. However, if you’re building without
  /// > standard library, this function will always return successfully. You have to take extra care
  /// > in this case.
  pub(crate) fn new() -> Result<Self, StateQueryError> {
    TLS_ACQUIRE_GFX_STATE.with(|rc| {
      let mut inner = rc.borrow_mut();

      match *inner {
        Some(_) => {
          inner.take();
          Self::get_from_context()
        }

        None => Err(StateQueryError::UnavailableGLState),
      }
    })
  }

  /// Get a `GraphicsContext` from the current OpenGL context.
  fn get_from_context() -> Result<Self, StateQueryError> {
    unsafe {
      let binding_stack = BindingStack::new();
      let viewport = Cached::new(get_ctx_viewport()?);
      let clear_color = Cached::new(get_ctx_clear_color()?);
      let clear_depth = Cached::new(get_ctx_clear_depth()?);
      let clear_stencil = Cached::new(get_ctx_clear_stencil()?);
      let blending_state = Cached::new(get_ctx_blending_state()?);
      let blending_equations = Cached::new(get_ctx_blending_equations()?);
      let blending_funcs = Cached::new(get_ctx_blending_factors()?);
      let depth_test = Cached::new(get_ctx_depth_test()?);
      let depth_test_comparison = Cached::new(Comparison::Less);
      let depth_write = Cached::new(get_ctx_depth_write()?);
      let stencil_test_enabled = Cached::new(get_ctx_stencil_test_enabled()?);
      let stencil_test = Cached::new(get_ctx_stencil_test()?);
      let stencil_operations = Cached::new(get_ctx_stencil_operations()?);
      let face_culling_state = Cached::new(get_ctx_face_culling_state()?);
      let face_culling_order = Cached::new(get_ctx_face_culling_order()?);
      let face_culling_mode = Cached::new(get_ctx_face_culling_mode()?);
      let vertex_restart = Cached::new(get_ctx_vertex_restart()?);
      let patch_vertex_nb = Cached::new(0);
      let current_texture_unit = Cached::new(get_ctx_current_texture_unit()?);
      let bound_textures = vec![(gl::TEXTURE_2D, 0); 48]; // 48 is the platform minimal requirement
      let texture_swimming_pool = Vec::new();
      let bound_uniform_buffers = vec![0; 36]; // 36 is the platform minimal requirement
      let bound_array_buffer = 0;
      let bound_element_array_buffer = 0;
      let bound_draw_framebuffer = Cached::new(get_ctx_bound_draw_framebuffer()?);
      let bound_vertex_array = get_ctx_bound_vertex_array()?;
      let current_program = get_ctx_current_program()?;
      let srgb_framebuffer_enabled = Cached::new(get_ctx_srgb_framebuffer_enabled()?);
      let scissor_state = Cached::new(get_ctx_scissor_state()?);
      let scissor_region = Cached::new(get_ctx_scissor_region()?);
      let vendor_name = None;
      let renderer_name = None;
      let gl_version = None;
      let glsl_version = None;
      let max_texture_array_elements = None;

      Ok(GLState {
        _a: PhantomData,
        binding_stack,
        viewport,
        clear_color,
        clear_depth,
        clear_stencil,
        blending_state,
        blending_equations,
        blending_funcs,
        depth_test,
        depth_test_comparison,
        depth_write,
        stencil_test_enabled,
        stencil_test,
        stencil_operations,
        face_culling_state,
        face_culling_order,
        face_culling_mode,
        vertex_restart,
        patch_vertex_nb,
        current_texture_unit,
        bound_textures,
        texture_swimming_pool,
        bound_uniform_buffers,
        bound_array_buffer,
        bound_element_array_buffer,
        bound_draw_framebuffer,
        bound_vertex_array,
        current_program,
        srgb_framebuffer_enabled,
        scissor_state,
        scissor_region,
        vendor_name,
        renderer_name,
        gl_version,
        glsl_version,
        max_texture_array_elements,
      })
    }
  }

  /// Invalidate the currently in-use vertex array.
  pub fn invalidate_vertex_array(&mut self) {
    self.bound_vertex_array = 0;
  }

  /// Invalidate the currently in-use array buffer.
  pub fn invalidate_array_buffer(&mut self) {
    self.bound_array_buffer = 0;
  }

  /// Invalidate the currently in-use shader program.
  pub fn invalidate_shader_program(&mut self) {
    self.current_program = 0;
  }

  /// Invalidate the currently in-use framebuffer.
  pub fn invalidate_framebuffer(&mut self) {
    self.bound_draw_framebuffer.invalidate();
  }

  /// Invalidate the currently in-use element array buffer.
  pub fn invalidate_element_array_buffer(&mut self) {
    self.bound_element_array_buffer = 0;
  }

  /// Invalidate the currently in-use texture unit.
  pub fn invalidate_texture_unit(&mut self) {
    self.current_texture_unit.invalidate();
  }

  /// Invalidate the texture bindings.
  pub fn invalidate_bound_textures(&mut self) {
    for t in &mut self.bound_textures {
      *t = (gl::TEXTURE_2D, 0);
    }
  }

  /// Invalidate the uniform buffer bindings.
  pub fn invalidate_bound_uniform_buffers(&mut self) {
    for b in &mut self.bound_uniform_buffers {
      *b = 0;
    }
  }

  /// Invalidate the currently in-use viewport.
  pub fn invalidate_viewport(&mut self) {
    self.viewport.invalidate()
  }

  /// Invalidate the currently in-use clear color.
  pub fn invalidate_clear_color(&mut self) {
    self.clear_color.invalidate()
  }

  /// Invalidate the currently in-use blending state.
  pub fn invalidate_blending_state(&mut self) {
    self.blending_state.invalidate()
  }

  /// Invalidate the currently in-use blending equation.
  pub fn invalidate_blending_equation(&mut self) {
    self.blending_equations.invalidate()
  }

  /// Invalidate the currently in-use blending function.
  pub fn invalidate_blending_func(&mut self) {
    self.blending_funcs.invalidate()
  }

  /// Invalidate the currently in-use depth test.
  pub fn invalidate_depth_test(&mut self) {
    self.depth_test.invalidate()
  }

  /// Invalidate the currently in-use depth test comparison.
  pub fn invalidate_depth_test_comparison(&mut self) {
    self.depth_test_comparison.invalidate()
  }

  /// Invalidate the currently in-use depth write state.
  pub fn invalidate_depth_write(&mut self) {
    self.depth_write.invalidate()
  }

  /// Invalidate the currently in-use face culling state.
  pub fn invalidate_face_culling_state(&mut self) {
    self.face_culling_state.invalidate()
  }

  /// Invalidate the currently in-use face culling order.
  pub fn invalidate_face_culling_order(&mut self) {
    self.face_culling_order.invalidate()
  }

  /// Invalidate the currently in-use face culling mode.
  pub fn invalidate_face_culling_mode(&mut self) {
    self.face_culling_mode.invalidate()
  }

  /// Invalidate the currently in-use vertex restart state.
  pub fn invalidate_vertex_restart(&mut self) {
    self.vertex_restart.invalidate()
  }

  /// Invalidate the currently in-use patch vertex number.
  pub fn invalidate_patch_vertex_nb(&mut self) {
    self.patch_vertex_nb.invalidate()
  }

  /// Invalidate the currently in-use sRGB framebuffer state.
  pub fn invalidate_srgb_framebuffer_enabled(&mut self) {
    self.srgb_framebuffer_enabled.invalidate()
  }

  /// Marshal a string represented as `*const c_uchar`, represented by the input argument, into a `&str`.
  ///
  /// The string is returned in a lossy way, which means that non-unicode characters go wheeeeeeeeeeee.
  fn marshal_gl_string(repr: GLenum) -> String {
    unsafe {
      let name_ptr = gl::GetString(repr);
      let name = CStr::from_ptr(name_ptr as *const c_char);
      name.to_string_lossy().into_owned()
    }
  }

  /// Get the OpenGL vendor name.
  ///
  /// Cache the name on the first call and then re-use it for later calls.
  pub fn get_vendor_name(&mut self) -> String {
    self.vendor_name.as_ref().cloned().unwrap_or_else(|| {
      let name = Self::marshal_gl_string(gl::VENDOR);
      self.vendor_name = Some(name.clone());
      name
    })
  }

  /// Get the OpenGL renderer name.
  ///
  /// Cache the name on the first call and then re-use it for later calls.
  pub fn get_renderer_name(&mut self) -> String {
    self.renderer_name.as_ref().cloned().unwrap_or_else(|| {
      let name = Self::marshal_gl_string(gl::RENDERER);
      self.renderer_name = Some(name.clone());
      name
    })
  }

  /// Get the OpenGL version.
  ///
  /// Cache the version on the first call and then re-use it for later calls.
  pub fn get_gl_version(&mut self) -> String {
    self.gl_version.as_ref().cloned().unwrap_or_else(|| {
      let version = Self::marshal_gl_string(gl::VERSION);
      self.gl_version = Some(version.clone());
      version
    })
  }

  /// Get the GLSL version.
  ///
  /// Cache the version on the first call and then re-use it for later calls.
  pub fn get_glsl_version(&mut self) -> String {
    self.glsl_version.as_ref().cloned().unwrap_or_else(|| {
      let version = Self::marshal_gl_string(gl::SHADING_LANGUAGE_VERSION);
      self.glsl_version = Some(version.clone());
      version
    })
  }

  /// Get the number of maximum elements an array texture can hold.
  ///
  /// Cache the number on the first call and then re-use it for later calls.
  pub fn get_max_texture_array_elements(&mut self) -> usize {
    self.max_texture_array_elements.unwrap_or_else(|| {
      let mut max = 0;
      unsafe { gl::GetIntegerv(gl::MAX_ARRAY_TEXTURE_LAYERS, &mut max) };
      let max = max as usize;
      self.max_texture_array_elements = Some(max);
      max
    })
  }

  pub(crate) fn binding_stack_mut(&mut self) -> &mut BindingStack {
    &mut self.binding_stack
  }

  pub(crate) fn create_texture(&mut self) -> GLuint {
    self.texture_swimming_pool.pop().unwrap_or_else(|| {
      let mut texture = 0;

      unsafe { gl::GenTextures(1, &mut texture) };
      texture
    })
  }

  /// Reserve at least a given number of textures.
  pub(crate) fn reserve_textures(&mut self, nb: usize) {
    let available = self.texture_swimming_pool.len();
    let needed = nb.max(available) - available;

    if needed > 0 {
      // resize the internal buffer to hold all the new textures and create a slice starting from
      // the previous end to the new end
      self.texture_swimming_pool.resize(available + needed, 0);
      let textures = &mut self.texture_swimming_pool[available..];

      unsafe { gl::GenTextures(needed as _, textures.as_mut_ptr()) };
    }
  }

  pub(crate) unsafe fn set_viewport(&mut self, viewport: [GLint; 4]) {
    if self.viewport.is_invalid(&viewport) {
      gl::Viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
      self.viewport.set(viewport);
    }
  }

  pub(crate) unsafe fn set_clear_color(&mut self, clear_color: [GLfloat; 4]) {
    if self.clear_color.is_invalid(&clear_color) {
      gl::ClearColor(
        clear_color[0],
        clear_color[1],
        clear_color[2],
        clear_color[3],
      );
      self.clear_color.set(clear_color);
    }
  }

  pub(crate) unsafe fn set_clear_depth(&mut self, clear_depth: GLfloat) {
    if self.clear_depth.is_invalid(&clear_depth) {
      gl::ClearDepth(clear_depth as _);
      self.clear_depth.set(clear_depth);
    }
  }

  pub(crate) unsafe fn set_clear_stencil(&mut self, clear_stencil: GLint) {
    if self.clear_stencil.is_invalid(&clear_stencil) {
      gl::ClearStencil(clear_stencil);
      self.clear_stencil.set(clear_stencil);
    }
  }

  pub(crate) unsafe fn set_blending_state(&mut self, state: BlendingState) {
    if self.blending_state.is_invalid(&state) {
      match state {
        BlendingState::On => gl::Enable(gl::BLEND),
        BlendingState::Off => gl::Disable(gl::BLEND),
      }

      self.blending_state.set(state);
    }
  }

  pub(crate) unsafe fn set_scissor_state(&mut self, state: ScissorState) {
    if self.scissor_state.is_invalid(&state) {
      match state {
        ScissorState::On => gl::Enable(gl::SCISSOR_TEST),
        ScissorState::Off => gl::Disable(gl::SCISSOR_TEST),
      }

      self.scissor_state.set(state);
    }
  }

  pub(crate) unsafe fn set_scissor_region(&mut self, region: &ScissorRegion) {
    if self.scissor_region.is_invalid(region) {
      let ScissorRegion {
        x,
        y,
        width,
        height,
      } = *region;

      gl::Scissor(x as GLint, y as GLint, width as GLint, height as GLint);

      self.scissor_region.set(*region);
    }
  }

  pub(crate) unsafe fn set_blending_equation(&mut self, equation: Equation) {
    let equations = BlendingEquations {
      rgb: equation,
      alpha: equation,
    };

    if self.blending_equations.is_invalid(&equations) {
      gl::BlendEquation(from_blending_equation(equation));
      self.blending_equations.set(equations);
    }
  }

  pub(crate) unsafe fn set_blending_equation_separate(
    &mut self,
    equation_rgb: Equation,
    equation_alpha: Equation,
  ) {
    let equations = BlendingEquations {
      rgb: equation_rgb,
      alpha: equation_alpha,
    };

    if self.blending_equations.is_invalid(&equations) {
      gl::BlendEquationSeparate(
        from_blending_equation(equation_rgb),
        from_blending_equation(equation_alpha),
      );

      self.blending_equations.set(equations);
    }
  }

  pub(crate) unsafe fn set_blending_func(&mut self, src: Factor, dst: Factor) {
    let funcs = BlendingFactors {
      src_rgb: src,
      dst_rgb: dst,
      src_alpha: src,
      dst_alpha: dst,
    };

    if self.blending_funcs.is_invalid(&funcs) {
      gl::BlendFunc(from_blending_factor(src), from_blending_factor(dst));
      self.blending_funcs.set(funcs);
    }
  }

  pub(crate) unsafe fn set_blending_func_separate(
    &mut self,
    src_rgb: Factor,
    dst_rgb: Factor,
    src_alpha: Factor,
    dst_alpha: Factor,
  ) {
    let funcs = BlendingFactors {
      src_rgb,
      dst_rgb,
      src_alpha,
      dst_alpha,
    };

    if self.blending_funcs.is_invalid(&funcs) {
      gl::BlendFuncSeparate(
        from_blending_factor(src_rgb),
        from_blending_factor(dst_rgb),
        from_blending_factor(src_alpha),
        from_blending_factor(dst_alpha),
      );

      self.blending_funcs.set(funcs);
    }
  }

  pub(crate) unsafe fn set_depth_test(&mut self, depth_test: DepthTest) {
    if self.depth_test.is_invalid(&depth_test) {
      match depth_test {
        DepthTest::On => gl::Enable(gl::DEPTH_TEST),
        DepthTest::Off => gl::Disable(gl::DEPTH_TEST),
      }

      self.depth_test.set(depth_test);
    }
  }

  pub(crate) unsafe fn set_depth_test_comparison(&mut self, depth_test_comparison: Comparison) {
    if self
      .depth_test_comparison
      .is_invalid(&depth_test_comparison)
    {
      gl::DepthFunc(comparison_to_glenum(depth_test_comparison));
      self.depth_test_comparison.set(depth_test_comparison);
    }
  }

  pub(crate) unsafe fn set_depth_write(&mut self, depth_write: Write) {
    if self.depth_write.is_invalid(&depth_write) {
      let enabled = match depth_write {
        Write::On => gl::TRUE,
        Write::Off => gl::FALSE,
      };

      gl::DepthMask(enabled);

      self.depth_write.set(depth_write);
    }
  }

  pub(crate) unsafe fn enable_stencil_test(&mut self, enable: bool) {
    if self.stencil_test_enabled.is_invalid(&enable) {
      if enable {
        gl::Enable(gl::STENCIL_TEST);
      } else {
        gl::Disable(gl::STENCIL_TEST);
      }

      self.stencil_test_enabled.set(enable);
    }
  }

  pub(crate) unsafe fn set_stencil_test(&mut self, stencil_test: StencilTest) {
    if self.stencil_test.is_invalid(&stencil_test) {
      let comparison = comparison_to_glenum(stencil_test.comparison);
      gl::StencilFunc(
        comparison,
        stencil_test.reference as _,
        stencil_test.mask as _,
      );

      self.stencil_test.set(stencil_test);
    }
  }

  pub(crate) unsafe fn set_stencil_operations(&mut self, stencil_ops: StencilOperations) {
    if self.stencil_operations.is_invalid(&stencil_ops) {
      gl::StencilOp(
        stencil_op_to_glenum(stencil_ops.depth_passes_stencil_fails),
        stencil_op_to_glenum(stencil_ops.depth_fails_stencil_passes),
        stencil_op_to_glenum(stencil_ops.depth_stencil_pass),
      );

      self.stencil_operations.set(stencil_ops);
    }
  }

  pub(crate) unsafe fn set_face_culling_state(&mut self, state: FaceCullingState) {
    if self.face_culling_state.is_invalid(&state) {
      match state {
        FaceCullingState::On => gl::Enable(gl::CULL_FACE),
        FaceCullingState::Off => gl::Disable(gl::CULL_FACE),
      }

      self.face_culling_state.set(state);
    }
  }

  pub(crate) unsafe fn set_face_culling_order(&mut self, order: FaceCullingOrder) {
    if self.face_culling_order.is_invalid(&order) {
      match order {
        FaceCullingOrder::CW => gl::FrontFace(gl::CW),
        FaceCullingOrder::CCW => gl::FrontFace(gl::CCW),
      }

      self.face_culling_order.set(order);
    }
  }

  pub(crate) unsafe fn set_face_culling_mode(&mut self, mode: FaceCullingMode) {
    if self.face_culling_mode.is_invalid(&mode) {
      match mode {
        FaceCullingMode::Front => gl::CullFace(gl::FRONT),
        FaceCullingMode::Back => gl::CullFace(gl::BACK),
        FaceCullingMode::Both => gl::CullFace(gl::FRONT_AND_BACK),
      }

      self.face_culling_mode.set(mode);
    }
  }

  pub(crate) unsafe fn set_vertex_restart(&mut self, state: VertexRestart) {
    if self.vertex_restart.is_invalid(&state) {
      match state {
        VertexRestart::On => gl::Enable(gl::PRIMITIVE_RESTART),
        VertexRestart::Off => gl::Disable(gl::PRIMITIVE_RESTART),
      }

      self.vertex_restart.set(state);
    }
  }

  pub(crate) unsafe fn set_patch_vertex_nb(&mut self, nb: usize) {
    if self.patch_vertex_nb.is_invalid(&nb) {
      gl::PatchParameteri(gl::PATCH_VERTICES, nb as GLint);
      self.patch_vertex_nb.set(nb);
    }
  }

  pub(crate) unsafe fn set_texture_unit(&mut self, unit: u32) {
    let unit = unit as GLenum;

    if self.current_texture_unit.is_invalid(&unit) {
      gl::ActiveTexture(gl::TEXTURE0 + unit);
      self.current_texture_unit.set(unit);
    }
  }

  /// Bind a texture at the current texture unit.
  pub unsafe fn bind_texture(&mut self, target: GLenum, handle: GLuint) {
    // Unwrap should be safe here, because we should always bind the texture unit before we bind the texture.
    // Maybe this should be handled differently?
    let unit = self.current_texture_unit.0.unwrap();
    self.bind_texture_at(target, handle, unit);
  }

  /// Bind the texture at the provided texture unit.
  pub(crate) unsafe fn bind_texture_at(&mut self, target: GLenum, handle: GLuint, unit: u32) {
    match self.bound_textures.get(unit as usize).cloned() {
      // there’s a abound texture, which is different from the one we want to bind
      Some((target_, handle_)) if target != target_ || handle != handle_ => {
        self.set_texture_unit(unit);
        gl::BindTexture(target, handle);
        self.bound_textures[unit as usize] = (target, handle);
      }

      // no texture bound at this unit; bind it
      None => {
        self.set_texture_unit(unit);
        gl::BindTexture(target, handle);

        // not enough registered texture units; let’s grow a bit more
        let unit = unit as usize;
        self.bound_textures.resize(unit + 1, (gl::TEXTURE_2D, 0));
        self.bound_textures[unit] = (target, handle);
      }

      _ => (), // cached
    }
  }

  pub(crate) unsafe fn bind_array_buffer(&mut self, handle: GLuint, bind: Bind) {
    if bind == Bind::Forced || self.bound_array_buffer != handle {
      gl::BindBuffer(gl::ARRAY_BUFFER, handle);
      self.bound_array_buffer = handle;
    }
  }

  pub(crate) unsafe fn bind_element_array_buffer(&mut self, handle: GLuint, bind: Bind) {
    if bind == Bind::Forced || self.bound_element_array_buffer != handle {
      gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, handle);
      self.bound_element_array_buffer = handle;
    }
  }

  pub(crate) unsafe fn bind_uniform_buffer(&mut self, handle: GLuint, binding: u32) {
    let binding_ = binding as usize;

    match self.bound_uniform_buffers.get(binding_) {
      Some(&handle_) if handle != handle_ => {
        gl::BindBufferBase(gl::UNIFORM_BUFFER, binding as GLuint, handle);
        self.bound_uniform_buffers[binding_] = handle;
      }

      None => {
        gl::BindBufferBase(gl::UNIFORM_BUFFER, binding as GLuint, handle);

        // not enough registered buffer bindings; let’s grow a bit more
        self.bound_uniform_buffers.resize(binding_ + 1, 0);
        self.bound_uniform_buffers[binding_] = handle;
      }

      _ => (), // cached
    }
  }

  pub(crate) unsafe fn unbind_buffer(&mut self, handle: GLuint) {
    if self.bound_array_buffer == handle {
      self.bind_array_buffer(0, Bind::Cached);
    } else if self.bound_element_array_buffer == handle {
      self.bind_element_array_buffer(0, Bind::Cached);
    } else if let Some(handle_) = self
      .bound_uniform_buffers
      .iter_mut()
      .find(|h| **h == handle)
    {
      *handle_ = 0;
    }
  }

  pub(crate) unsafe fn bind_draw_framebuffer(&mut self, handle: GLuint) {
    if self.bound_draw_framebuffer.is_invalid(&handle) {
      gl::BindFramebuffer(gl::DRAW_FRAMEBUFFER, handle);
      self.bound_draw_framebuffer.set(handle);
    }
  }

  pub(crate) unsafe fn bind_vertex_array(&mut self, handle: GLuint, bind: Bind) {
    if bind == Bind::Forced || self.bound_vertex_array != handle {
      gl::BindVertexArray(handle);
      self.bound_vertex_array = handle;
    }
  }

  pub(crate) unsafe fn unbind_vertex_array(&mut self) {
    self.bind_vertex_array(0, Bind::Cached)
  }

  pub(crate) unsafe fn use_program(&mut self, handle: GLuint) {
    if self.current_program != handle {
      gl::UseProgram(handle);
      self.current_program = handle;
    }
  }

  pub(crate) unsafe fn enable_srgb_framebuffer(&mut self, srgb_framebuffer_enabled: bool) {
    if self
      .srgb_framebuffer_enabled
      .is_invalid(&srgb_framebuffer_enabled)
    {
      if srgb_framebuffer_enabled {
        gl::Enable(gl::FRAMEBUFFER_SRGB);
      } else {
        gl::Disable(gl::FRAMEBUFFER_SRGB);
      }

      self.srgb_framebuffer_enabled.set(srgb_framebuffer_enabled);
    }
  }
}

/// Should the binding be cached or forced to the provided value?
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Bind {
  Forced,
  Cached,
}

#[inline]
fn from_blending_equation(equation: Equation) -> GLenum {
  match equation {
    Equation::Additive => gl::FUNC_ADD,
    Equation::Subtract => gl::FUNC_SUBTRACT,
    Equation::ReverseSubtract => gl::FUNC_REVERSE_SUBTRACT,
    Equation::Min => gl::MIN,
    Equation::Max => gl::MAX,
  }
}

#[inline]
fn from_blending_factor(factor: Factor) -> GLenum {
  match factor {
    Factor::One => gl::ONE,
    Factor::Zero => gl::ZERO,
    Factor::SrcColor => gl::SRC_COLOR,
    Factor::SrcColorComplement => gl::ONE_MINUS_SRC_COLOR,
    Factor::DestColor => gl::DST_COLOR,
    Factor::DestColorComplement => gl::ONE_MINUS_DST_COLOR,
    Factor::SrcAlpha => gl::SRC_ALPHA,
    Factor::SrcAlphaComplement => gl::ONE_MINUS_SRC_ALPHA,
    Factor::DstAlpha => gl::DST_ALPHA,
    Factor::DstAlphaComplement => gl::ONE_MINUS_DST_ALPHA,
    Factor::SrcAlphaSaturate => gl::SRC_ALPHA_SATURATE,
  }
}

/// An error that might happen when the context is queried.
#[non_exhaustive]
#[derive(Debug)]
pub enum StateQueryError {
  /// The [`GLState`] object is unavailable.
  ///
  /// That might occur if the current thread doesn’t support allocating a new graphics state. It
  /// might happen if you try to have more than one state on the same thread, for instance.
  UnavailableGLState,
  /// Corrupted blending state.
  UnknownBlendingState(GLboolean),
  /// Corrupted blending equation.
  UnknownBlendingEquation(GLenum),
  /// Corrupted blending source factor.
  UnknownBlendingSrcFactor(GLenum),
  /// Corrupted blending destination factor.
  UnknownBlendingDstFactor(GLenum),
  /// Corrupted depth test state.
  UnknownDepthTestState(GLboolean),
  /// Corrupted stencil test state.
  UnknownStencilTestState(GLboolean),
  /// Corrupted stencil test comparison.
  UnknownStencilTestComparison(GLint),
  /// Corrupted stencil op.
  UnknownStencilOp(GLint),
  /// Corrupted depth write state.
  UnknownWriteState(GLboolean),
  /// Corrupted face culling state.
  UnknownFaceCullingState(GLboolean),
  /// Corrupted face culling order.
  UnknownFaceCullingOrder(GLenum),
  /// Corrupted face culling mode.
  UnknownFaceCullingMode(GLenum),
  /// Corrupted vertex restart state.
  UnknownVertexRestartState(GLboolean),
  /// Corrupted sRGB framebuffer state.
  UnknownSRGBFramebufferState(GLboolean),
  /// Corrupted scissor state.
  UnknownScissorState(GLboolean),
}

impl fmt::Display for StateQueryError {
  fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
    match *self {
      StateQueryError::UnavailableGLState => write!(f, "unavailable graphics state"),
      StateQueryError::UnknownBlendingState(ref s) => write!(f, "unknown blending state: {}", s),
      StateQueryError::UnknownBlendingEquation(ref e) => {
        write!(f, "unknown blending equation: {}", e)
      }
      StateQueryError::UnknownBlendingSrcFactor(ref k) => {
        write!(f, "unknown blending source factor: {}", k)
      }
      StateQueryError::UnknownBlendingDstFactor(ref k) => {
        write!(f, "unknown blending destination factor: {}", k)
      }
      StateQueryError::UnknownDepthTestState(ref s) => write!(f, "unknown depth test state: {}", s),
      StateQueryError::UnknownWriteState(ref s) => {
        write!(f, "unknown depth write state: {}", s)
      }
      StateQueryError::UnknownStencilTestState(ref s) => {
        write!(f, "unknown stencil test state: {}", s)
      }
      StateQueryError::UnknownStencilTestComparison(ref k) => {
        write!(f, "unknown stencil test comparison: {}", k)
      }
      StateQueryError::UnknownStencilOp(ref op) => {
        write!(f, "unknown stencil operation: {}", op)
      }
      StateQueryError::UnknownFaceCullingState(ref s) => {
        write!(f, "unknown face culling state: {}", s)
      }
      StateQueryError::UnknownFaceCullingOrder(ref o) => {
        write!(f, "unknown face culling order: {}", o)
      }
      StateQueryError::UnknownFaceCullingMode(ref m) => {
        write!(f, "unknown face culling mode: {}", m)
      }
      StateQueryError::UnknownVertexRestartState(ref s) => {
        write!(f, "unknown vertex restart state: {}", s)
      }
      StateQueryError::UnknownSRGBFramebufferState(ref s) => {
        write!(f, "unknown sRGB framebuffer state: {}", s)
      }
      StateQueryError::UnknownScissorState(ref s) => write!(f, "unknown scissor state: {}", s),
    }
  }
}

impl error::Error for StateQueryError {}

unsafe fn get_ctx_viewport() -> Result<[GLint; 4], StateQueryError> {
  let mut data = [0; 4];
  gl::GetIntegerv(gl::VIEWPORT, data.as_mut_ptr());
  Ok(data)
}

unsafe fn get_ctx_clear_color() -> Result<[GLfloat; 4], StateQueryError> {
  let mut data = [0.; 4];
  gl::GetFloatv(gl::COLOR_CLEAR_VALUE, data.as_mut_ptr());
  Ok(data)
}

unsafe fn get_ctx_clear_depth() -> Result<GLfloat, StateQueryError> {
  let mut data = 0.;
  gl::GetFloatv(gl::DEPTH_CLEAR_VALUE, &mut data);
  Ok(data)
}

unsafe fn get_ctx_clear_stencil() -> Result<GLint, StateQueryError> {
  let mut data = 0;
  gl::GetIntegerv(gl::STENCIL_CLEAR_VALUE, &mut data);
  Ok(data)
}

unsafe fn get_ctx_blending_state() -> Result<BlendingState, StateQueryError> {
  let state = gl::IsEnabled(gl::BLEND);

  match state {
    gl::TRUE => Ok(BlendingState::On),
    gl::FALSE => Ok(BlendingState::Off),
    _ => Err(StateQueryError::UnknownBlendingState(state)),
  }
}

unsafe fn get_ctx_scissor_state() -> Result<ScissorState, StateQueryError> {
  let state = gl::IsEnabled(gl::SCISSOR_TEST);

  match state {
    gl::TRUE => Ok(ScissorState::On),
    gl::FALSE => Ok(ScissorState::Off),
    _ => Err(StateQueryError::UnknownScissorState(state)),
  }
}

unsafe fn get_ctx_scissor_region() -> Result<ScissorRegion, StateQueryError> {
  let mut data = [0; 4];
  gl::GetIntegerv(gl::SCISSOR_BOX, data.as_mut_ptr());

  Ok(ScissorRegion {
    x: data[0] as u32,
    y: data[1] as u32,
    width: data[2] as u32,
    height: data[3] as u32,
  })
}

unsafe fn get_ctx_blending_equations() -> Result<BlendingEquations, StateQueryError> {
  let mut rgb = gl::FUNC_ADD as GLint;
  let mut alpha = gl::FUNC_ADD as GLint;

  gl::GetIntegerv(gl::BLEND_EQUATION_RGB, &mut rgb);
  gl::GetIntegerv(gl::BLEND_EQUATION_ALPHA, &mut alpha);

  let rgb = map_enum_to_blending_equation(rgb as GLenum)?;
  let alpha = map_enum_to_blending_equation(alpha as GLenum)?;

  Ok(BlendingEquations { rgb, alpha })
}

unsafe fn get_ctx_blending_factors() -> Result<BlendingFactors, StateQueryError> {
  let mut src_rgb = gl::ONE as GLint;
  let mut dst_rgb = gl::ZERO as GLint;
  let mut src_alpha = gl::ONE as GLint;
  let mut dst_alpha = gl::ZERO as GLint;

  gl::GetIntegerv(gl::BLEND_SRC_RGB, &mut src_rgb);
  gl::GetIntegerv(gl::BLEND_DST_RGB, &mut dst_rgb);
  gl::GetIntegerv(gl::BLEND_SRC_ALPHA, &mut src_alpha);
  gl::GetIntegerv(gl::BLEND_DST_ALPHA, &mut dst_alpha);

  let src_rgb = from_gl_blending_factor(src_rgb as GLenum)
    .map_err(StateQueryError::UnknownBlendingSrcFactor)?;
  let dst_rgb = from_gl_blending_factor(dst_rgb as GLenum)
    .map_err(StateQueryError::UnknownBlendingDstFactor)?;
  let src_alpha = from_gl_blending_factor(src_alpha as GLenum)
    .map_err(StateQueryError::UnknownBlendingSrcFactor)?;
  let dst_alpha = from_gl_blending_factor(dst_alpha as GLenum)
    .map_err(StateQueryError::UnknownBlendingDstFactor)?;

  Ok(BlendingFactors {
    src_rgb,
    dst_rgb,
    src_alpha,
    dst_alpha,
  })
}

#[inline]
fn map_enum_to_blending_equation(data: GLenum) -> Result<Equation, StateQueryError> {
  match data {
    gl::FUNC_ADD => Ok(Equation::Additive),
    gl::FUNC_SUBTRACT => Ok(Equation::Subtract),
    gl::FUNC_REVERSE_SUBTRACT => Ok(Equation::ReverseSubtract),
    gl::MIN => Ok(Equation::Min),
    gl::MAX => Ok(Equation::Max),
    _ => Err(StateQueryError::UnknownBlendingEquation(data)),
  }
}

#[inline]
fn from_gl_blending_factor(factor: GLenum) -> Result<Factor, GLenum> {
  match factor {
    gl::ONE => Ok(Factor::One),
    gl::ZERO => Ok(Factor::Zero),
    gl::SRC_COLOR => Ok(Factor::SrcColor),
    gl::ONE_MINUS_SRC_COLOR => Ok(Factor::SrcColorComplement),
    gl::DST_COLOR => Ok(Factor::DestColor),
    gl::ONE_MINUS_DST_COLOR => Ok(Factor::DestColorComplement),
    gl::SRC_ALPHA => Ok(Factor::SrcAlpha),
    gl::ONE_MINUS_SRC_ALPHA => Ok(Factor::SrcAlphaComplement),
    gl::DST_ALPHA => Ok(Factor::DstAlpha),
    gl::ONE_MINUS_DST_ALPHA => Ok(Factor::DstAlphaComplement),
    gl::SRC_ALPHA_SATURATE => Ok(Factor::SrcAlphaSaturate),
    _ => Err(factor),
  }
}

unsafe fn get_ctx_depth_test() -> Result<DepthTest, StateQueryError> {
  let state = gl::IsEnabled(gl::DEPTH_TEST);

  match state {
    gl::TRUE => Ok(DepthTest::On),
    gl::FALSE => Ok(DepthTest::Off),
    _ => Err(StateQueryError::UnknownDepthTestState(state)),
  }
}

unsafe fn get_ctx_depth_write() -> Result<Write, StateQueryError> {
  let mut state = gl::FALSE;

  gl::GetBooleanv(gl::DEPTH_WRITEMASK, &mut state);

  match state {
    gl::TRUE => Ok(Write::On),
    gl::FALSE => Ok(Write::Off),
    _ => Err(StateQueryError::UnknownWriteState(state)),
  }
}

unsafe fn get_ctx_stencil_test_enabled() -> Result<bool, StateQueryError> {
  let state = gl::IsEnabled(gl::STENCIL_TEST);

  match state {
    gl::TRUE => Ok(true),
    gl::FALSE => Ok(false),
    _ => Err(StateQueryError::UnknownStencilTestState(state)),
  }
}

unsafe fn get_ctx_stencil_test() -> Result<StencilTest, StateQueryError> {
  // we need the comparison function, the reference value and the mask
  let mut data = gl::ALWAYS as GLint;

  gl::GetIntegerv(gl::STENCIL_FUNC, &mut data);
  let comparison = glenum_to_comparison(data as GLenum)
    .ok_or_else(|| StateQueryError::UnknownStencilTestComparison(data))?;

  gl::GetIntegerv(gl::STENCIL_REF, &mut data);
  let reference = data as u8;

  gl::GetIntegerv(gl::STENCIL_VALUE_MASK, &mut data);
  let mask = data as u8;

  Ok(StencilTest {
    comparison,
    reference,
    mask,
  })
}

unsafe fn get_ctx_stencil_operations() -> Result<StencilOperations, StateQueryError> {
  let mut data = 0 as GLint;

  gl::GetIntegerv(gl::STENCIL_FAIL, &mut data);
  let depth_passes_stencil_fails =
    glenum_to_stencil_op(data as _).ok_or_else(|| StateQueryError::UnknownStencilOp(data))?;
  gl::GetIntegerv(gl::STENCIL_PASS_DEPTH_FAIL, &mut data);
  let depth_fails_stencil_passes =
    glenum_to_stencil_op(data as _).ok_or_else(|| StateQueryError::UnknownStencilOp(data))?;
  gl::GetIntegerv(gl::STENCIL_PASS_DEPTH_PASS, &mut data);
  let depth_stencil_pass =
    glenum_to_stencil_op(data as _).ok_or_else(|| StateQueryError::UnknownStencilOp(data))?;

  Ok(StencilOperations {
    depth_passes_stencil_fails,
    depth_fails_stencil_passes,
    depth_stencil_pass,
  })
}

unsafe fn get_ctx_face_culling_state() -> Result<FaceCullingState, StateQueryError> {
  let state = gl::IsEnabled(gl::CULL_FACE);

  match state {
    gl::TRUE => Ok(FaceCullingState::On),
    gl::FALSE => Ok(FaceCullingState::Off),
    _ => Err(StateQueryError::UnknownFaceCullingState(state)),
  }
}

unsafe fn get_ctx_face_culling_order() -> Result<FaceCullingOrder, StateQueryError> {
  let mut order = gl::CCW as GLint;
  gl::GetIntegerv(gl::FRONT_FACE, &mut order);

  let order = order as GLenum;
  match order {
    gl::CCW => Ok(FaceCullingOrder::CCW),
    gl::CW => Ok(FaceCullingOrder::CW),
    _ => Err(StateQueryError::UnknownFaceCullingOrder(order)),
  }
}

unsafe fn get_ctx_face_culling_mode() -> Result<FaceCullingMode, StateQueryError> {
  let mut mode = gl::BACK as GLint;
  gl::GetIntegerv(gl::CULL_FACE_MODE, &mut mode);

  let mode = mode as GLenum;
  match mode {
    gl::FRONT => Ok(FaceCullingMode::Front),
    gl::BACK => Ok(FaceCullingMode::Back),
    gl::FRONT_AND_BACK => Ok(FaceCullingMode::Both),
    _ => Err(StateQueryError::UnknownFaceCullingMode(mode)),
  }
}

unsafe fn get_ctx_vertex_restart() -> Result<VertexRestart, StateQueryError> {
  let state = gl::IsEnabled(gl::PRIMITIVE_RESTART);

  match state {
    gl::TRUE => Ok(VertexRestart::On),
    gl::FALSE => Ok(VertexRestart::Off),
    _ => Err(StateQueryError::UnknownVertexRestartState(state)),
  }
}

unsafe fn get_ctx_current_texture_unit() -> Result<GLenum, StateQueryError> {
  let mut active_texture = gl::TEXTURE0 as GLint;
  gl::GetIntegerv(gl::ACTIVE_TEXTURE, &mut active_texture);
  Ok(active_texture as GLenum)
}

unsafe fn get_ctx_bound_draw_framebuffer() -> Result<GLuint, StateQueryError> {
  let mut bound = 0 as GLint;
  gl::GetIntegerv(gl::DRAW_FRAMEBUFFER_BINDING, &mut bound);
  Ok(bound as GLuint)
}

unsafe fn get_ctx_bound_vertex_array() -> Result<GLuint, StateQueryError> {
  let mut bound = 0 as GLint;
  gl::GetIntegerv(gl::VERTEX_ARRAY_BINDING, &mut bound);
  Ok(bound as GLuint)
}

unsafe fn get_ctx_current_program() -> Result<GLuint, StateQueryError> {
  let mut used = 0 as GLint;
  gl::GetIntegerv(gl::CURRENT_PROGRAM, &mut used);
  Ok(used as GLuint)
}

unsafe fn get_ctx_srgb_framebuffer_enabled() -> Result<bool, StateQueryError> {
  let state = gl::IsEnabled(gl::FRAMEBUFFER_SRGB);

  match state {
    gl::TRUE => Ok(true),
    gl::FALSE => Ok(false),
    _ => Err(StateQueryError::UnknownSRGBFramebufferState(state)),
  }
}

/// Whether or not enable blending.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum BlendingState {
  /// Enable blending.
  On,
  /// Disable blending.
  Off,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BlendingFactors {
  src_rgb: Factor,
  dst_rgb: Factor,
  src_alpha: Factor,
  dst_alpha: Factor,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BlendingEquations {
  rgb: Equation,
  alpha: Equation,
}

/// Whether or not depth test should be enabled.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum DepthTest {
  /// The depth test is enabled.
  On,
  /// The depth test is disabled.
  Off,
}

/// Should face culling be enabled?
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum FaceCullingState {
  /// Enable face culling.
  On,
  /// Disable face culling.
  Off,
}

/// Whether or not enable scissor test.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum ScissorState {
  /// Enable scissor.
  On,
  /// Disable scissor.
  Off,
}
