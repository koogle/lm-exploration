const canvas = document.querySelector('#canvas');
const errorElement = document.querySelector('#error');
const controls = document.querySelector('#controls');
let animationId;

function showError(message) {
  cancelAnimationFrame(animationId);
  errorElement.textContent = message;
  errorElement.hidden = false;
  controls.disabled = true;
}

async function start() {
  if (!navigator.gpu) throw new Error('This demo needs a browser with WebGPU enabled.');
  const response = await fetch('./inference.wasm', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load the Rust library (${response.status}). Run make build.`);
  const { instance } = await WebAssembly.instantiateStreaming(response, {});
  const rust = instance.exports;
  const shader = new TextDecoder().decode(
    new Uint8Array(rust.memory.buffer, rust.shader_ptr(), rust.shader_len()),
  );

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter is available.');
  const device = await adapter.requestDevice();
  device.lost.then(({ message }) => showError(`GPU device lost. Refresh to restart. ${message}`));
  device.addEventListener('uncapturederror', (event) => showError(event.error.message));
  const context = canvas.getContext('webgpu');
  if (!context) throw new Error('Could not create a WebGPU canvas.');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const module = device.createShaderModule({ code: shader });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === 'error');
  if (errors.length) throw new Error(errors.map((error) => `${error.lineNum}: ${error.message}`).join('\n'));
  const pipeline = await device.createRenderPipelineAsync({
    layout: 'auto',
    vertex: { module, entryPoint: 'vertex_main' },
    fragment: { module, entryPoint: 'fragment_main', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const uniformSize = rust.uniform_size();
  const uniforms = device.createBuffer({ size: uniformSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniforms } }],
  });
  const sliders = [
    ['zoom-level', 'zoom-value', rust.set_zoom_level, (value) => `${(2 ** value).toLocaleString(undefined, { maximumSignificantDigits: 3, notation: Math.abs(value) > 16 ? 'scientific' : 'standard' })}×`],
    ['iterations', 'iterations-value', rust.set_iterations, (value) => Math.round(value)],
    ['color-density', 'density-value', rust.set_color_density, (value) => value.toFixed(3)],
    ['color-speed', 'speed-value', rust.set_color_speed, (value) => `${value.toFixed(3)}/s`],
  ].map(([inputId, outputId, set, format]) => ({
    input: document.getElementById(inputId), output: document.getElementById(outputId), set, format,
  }));
  function syncSliders() {
    const values = new Float32Array(rust.memory.buffer, rust.read_settings(), sliders.length);
    sliders.forEach(({ input, output, format }, index) => {
      // Extend the slider window near either end instead of capping the view.
      if (index === 0) {
        if (values[index] < Number(input.min) + 1) input.min = Math.floor(values[index] - 8);
        if (values[index] > Number(input.max) - 1) input.max = Math.ceil(values[index] + 8);
      }
      input.value = values[index];
      output.value = format(values[index]);
    });
  }
  sliders.forEach(({ input, set }) => input.addEventListener('input', () => {
    set(Number(input.value));
    syncSliders();
  }));
  syncSliders();
  const pauseButton = controls.querySelector('[data-action="pause"]');
  function updatePauseButton(paused) {
    pauseButton.textContent = paused ? 'Resume colors' : 'Pause colors';
    pauseButton.setAttribute('aria-pressed', String(Boolean(paused)));
  }
  controls.addEventListener('click', (event) => {
    const action = event.target.closest('button')?.dataset.action;
    if (!action) return;
    switch (action) {
      case 'left': rust.pan(-1, 0); break;
      case 'right': rust.pan(1, 0); break;
      case 'up': rust.pan(0, 1); break;
      case 'down': rust.pan(0, -1); break;
      case 'in': rust.zoom(0.7); break;
      case 'out': rust.zoom(1 / 0.7); break;
      case 'detail': rust.explore_detail(); break;
      case 'pause': updatePauseButton(rust.toggle_pause()); break;
      case 'reset': rust.reset_view(); updatePauseButton(false); break;
    }
    syncSliders();
  });
  controls.disabled = false;
  let drag;
  function endDrag() {
    const pointerId = drag?.pointerId;
    drag = undefined;
    canvas.classList.remove('dragging');
    if (pointerId !== undefined && canvas.hasPointerCapture(pointerId)) {
      canvas.releasePointerCapture(pointerId);
    }
  }
  canvas.addEventListener('pointerdown', (event) => {
    if (controls.disabled || event.button !== 0 || drag) return;
    event.preventDefault();
    drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add('dragging');
  });
  canvas.addEventListener('pointermove', (event) => {
    if (controls.disabled || event.pointerId !== drag?.pointerId) return;
    const { height } = canvas.getBoundingClientRect();
    if (!height) return;
    // Move the image with the pointer, converting CSS pixels to shader space.
    rust.drag(-2 * (event.clientX - drag.x) / height, 2 * (event.clientY - drag.y) / height);
    drag.x = event.clientX;
    drag.y = event.clientY;
  });
  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    canvas.addEventListener(name, (event) => {
      if (event.pointerId === drag?.pointerId) endDrag();
    });
  }
  window.addEventListener('blur', endDrag);
  canvas.addEventListener('wheel', (event) => {
    if (controls.disabled) return;
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    if (!rect.height) return;
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? rect.height : 1;
    // Trackpad pinch arrives as Ctrl+wheel; normal two-finger scroll also zooms.
    const sensitivity = event.ctrlKey ? 0.01 : 0.002;
    const factor = Math.exp(Math.max(-0.5, Math.min(0.5, event.deltaY * unit * sensitivity)));
    rust.zoom_at(factor,
      (2 * (event.clientX - rect.left) - rect.width) / rect.height,
      (rect.height - 2 * (event.clientY - rect.top)) / rect.height);
    syncSliders();
  }, { passive: false });
  const startTime = performance.now();

  function render(now) {
    try {
      // Bound fractal-rendering cost on high-density displays and large windows.
      const scale = Math.min(devicePixelRatio, 1.5, 1600 / Math.max(innerWidth, innerHeight));
      const width = Math.min(device.limits.maxTextureDimension2D, Math.max(1, Math.round(innerWidth * scale)));
      const height = Math.min(device.limits.maxTextureDimension2D, Math.max(1, Math.round(innerHeight * scale)));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const texture = context.getCurrentTexture();
      const pointer = rust.update_frame(texture.width, texture.height, (now - startTime) / 1000);
      device.queue.writeBuffer(uniforms, 0, new Uint8Array(rust.memory.buffer, pointer, uniformSize));
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: texture.createView(),
          loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
      animationId = requestAnimationFrame(render);
    } catch (error) {
      showError(error.message);
    }
  }
  animationId = requestAnimationFrame(render);
}

start().catch((error) => showError(error.message));
