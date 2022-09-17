use crate::{
  backend::Backend,
  dim::Dimensionable,
  framebuffer::Framebuffer,
  pipeline::{PipelineState, WithFramebuffer},
  primitive::Primitive,
  render_slots::{DepthRenderSlot, RenderSlots},
  shader::{FromEnv, Program, ProgramBuilder, ProgramUpdate},
  vertex::Vertex,
  vertex_entity::{Indices, VertexEntity, Vertices},
  vertex_storage::VertexStorage,
};

#[derive(Debug)]
pub struct Context<B> {
  backend: B,
}

impl<B> Context<B>
where
  B: Backend,
{
  pub unsafe fn new(backend: B) -> Self {
    Self { backend }
  }

  pub fn new_vertex_entity<V, S, I>(
    &mut self,
    storage: S,
    indices: I,
  ) -> Result<VertexEntity<V, S>, B::Err>
  where
    V: Vertex,
    S: VertexStorage<V>,
    I: Into<Vec<u32>>,
  {
    unsafe { self.backend.new_vertex_entity(storage, indices) }
  }

  pub fn vertices<'a, V, S>(
    &'a mut self,
    entity: &VertexEntity<V, S>,
  ) -> Result<Vertices<'a, V, S>, B::Err>
  where
    V: Vertex,
    S: VertexStorage<V>,
  {
    unsafe { self.backend.vertex_entity_vertices(entity) }
  }

  pub fn update_vertices<'a, V, S>(
    &'a mut self,
    entity: &VertexEntity<V, S>,
    vertices: Vertices<'a, V, S>,
  ) -> Result<(), B::Err>
  where
    V: Vertex,
    S: VertexStorage<V>,
  {
    unsafe { self.backend.vertex_entity_update_vertices(entity, vertices) }
  }

  pub fn indices<'a, V, S>(&'a mut self, entity: &VertexEntity<V, S>) -> Result<Indices<'a>, B::Err>
  where
    V: Vertex,
    S: VertexStorage<V>,
  {
    unsafe { self.backend.vertex_entity_indices(entity) }
  }

  pub fn update_indices<'a, V, S>(
    &'a mut self,
    entity: &VertexEntity<V, S>,
    indices: Indices<'a>,
  ) -> Result<(), B::Err>
  where
    V: Vertex,
    S: VertexStorage<V>,
  {
    unsafe { self.backend.vertex_entity_update_indices(entity, indices) }
  }

  pub fn new_framebuffer<D, RS, DS>(
    &mut self,
    size: D::Size,
  ) -> Result<Framebuffer<D, RS, DS>, B::Err>
  where
    D: Dimensionable,
    RS: RenderSlots,
    DS: DepthRenderSlot,
  {
    unsafe { self.backend.new_framebuffer(size) }
  }

  pub fn new_program<V, W, P, Q, S, E>(
    &mut self,
    builder: ProgramBuilder<V, W, P, Q, S, E>,
  ) -> Result<Program<V, P, S, E>, B::Err>
  where
    V: Vertex,
    W: Vertex,
    P: Primitive<Vertex = W>,
    Q: Primitive,
    S: RenderSlots,
    E: FromEnv,
  {
    unsafe {
      self.backend.new_program(
        builder.vertex_code,
        builder.primitive_code,
        builder.shading_code,
      )
    }
  }

  pub fn update_program<'a, V, P, S, E>(
    &'a mut self,
    program: &Program<V, P, S, E>,
    updater: impl FnOnce(ProgramUpdate<'a, B>, &E) -> Result<(), B::Err>,
  ) -> Result<(), B::Err> {
    let program_update = ProgramUpdate {
      backend: &mut self.backend,
      program_handle: program.handle(),
    };

    updater(program_update, &program.environment)
  }

  pub fn with_framebuffer<'a, D, CS, DS, Err>(
    &'a mut self,
    framebuffer: &Framebuffer<D, CS, DS>,
    state: &PipelineState,
    f: impl FnOnce(WithFramebuffer<'a, B, CS>) -> Result<(), Err>,
  ) -> Result<(), B::Err>
  where
    D: Dimensionable,
    CS: RenderSlots,
    DS: DepthRenderSlot,
    Err: From<B::Err>,
  {
    unsafe { self.backend.with_framebuffer(framebuffer, state, f) }
  }
}
