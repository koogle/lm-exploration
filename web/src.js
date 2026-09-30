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
  const pauseButton = controls.querySelector('[data-action="pause"]');
  function updatePauseButton(paused) {
    pauseButton.textContent = paused ? 'Resume colors' : 'Pause colors';
    pauseButton.setAttribute('aria-pressed', String(Boolean(paused)));
  }
  controls.addEventListener('click', (event) => {
    const action = event.target.closest('button')?.dataset.action;
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
  });
  controls.disabled = false;
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
