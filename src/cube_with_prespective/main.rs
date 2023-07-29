use libvktypes::{
    window,
    libvk,
    layers,
    extensions,
    surface,
    hw,
    dev,
    swapchain,
    memory,
    shader,
    graphics,
    sync,
    cmd,
    queue
};

const VERT_SHADER: &str = "
#version 460

layout(location = 0) in vec4 position;

layout(set = 0, binding = 0) uniform Transformations {
    mat4 camera;
    mat4 projection;
    mat4 scale;
    mat4 z_rotation;
    mat4 y_rotation;
} transformations;

void main() {
    vec4 projection =
        transformations.scale
        *transformations.projection
        *transformations.camera
        *transformations.y_rotation
        *transformations.z_rotation
        *transformations.scale
        *position;

    gl_Position = projection/projection.w;

    gl_Position.z += 1.0;
}
";

const FRAG_SHADER: &str = "
#version 460

layout(location = 0) out vec4 color;

layout(set = 0, binding = 1) uniform Colordata {
    vec4 data[6];
} colordata;

void main(){
    color = colordata.data[gl_PrimitiveID/2];
}
";

const VERTEX_DATA: &[f32] = &[
    -1.0, -1.0, -1.0, 1.0,
    -1.0, -1.0,  1.0, 1.0,

    1.0, -1.0, -1.0, 1.0,
    1.0, -1.0,  1.0, 1.0,

    -1.0, 1.0, -1.0, 1.0,
    -1.0, 1.0,  1.0, 1.0,

    1.0, 1.0, -1.0,  1.0,
    1.0, 1.0,  1.0,  1.0,
];

const INDICES: &[u32] = &[
    0, 5, 1,
    0, 4, 5,

    2, 0, 1,
    2, 1, 3,

    6, 5, 4,
    6, 7, 5,

    2, 3, 7,
    2, 7, 6,

    3, 5, 7,
    3, 1, 5,

    4, 2, 6,
    4, 0, 2,
];

/*
const VERTEX_DATA: &[f32] = &[
    1.0,  0.5, -1.0, 1.0,
    1.0, -0.5, -1.0, 1.0,

    0.0,  0.5, 0.0, 1.0,
    0.0, -0.5, 0.0, 1.0,
];

const INDICES: &[u32] = &[
    0, 1, 3,
    0, 3, 2
];
*/

const Y_ANGLE: f32 = std::f32::consts::FRAC_PI_4;

const COLOR_DATA: &[f32] = &[
    1.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 1.0,
    0.5, 0.0, 1.0, 1.0,
    1.0, 0.5, 1.0, 1.0,
    1.0, 1.0, 0.5, 1.0,
];

fn main() {
    let mut z_angle: f32 = 0.0;

    let mut transformations = [
        0.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, -2.0, 1.0,

        -1.0, 0.0, 0.0,   0.0,
        0.0, -1.0, 0.0,   0.0,
        0.0, 0.0, -0.25, -1.0,
        0.0, 0.0, -1.25,  0.0,

        0.5, 0.0, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.0, 1.0,

        // z rotation
        z_angle.cos(),  z_angle.sin(), 0.0, 0.0,
        -z_angle.sin(), z_angle.cos(), 0.0, 0.0,
        0.0,            0.0,           1.0, 0.0,
        0.0,            0.0,           0.0, 1.0,

        // y rotation
        Y_ANGLE.cos(), 0.0, -Y_ANGLE.sin(), 0.0,
        0.0,             1.0, 0.0,              0.0,
        Y_ANGLE.sin(), 0.0, Y_ANGLE.cos(),  0.0,
        0.0,             0.0, 0.0,              1.0
    ];

    let event_loop = window::eventloop();

    let wnd = window::create_window(&event_loop).expect("Failed to create window");

    let lib_type = libvk::InstanceType {
        debug_layer: Some(layers::DebugLayer::default()),
        extensions: &[extensions::DEBUG_EXT_NAME,
            extensions::SURFACE_EXT_NAME,
            extensions::XLIB_SURFACE_EXT_NAME],
        ..libvk::InstanceType::default()
    };

    let lib = libvk::Instance::new(&lib_type).expect("Failed to load library");

    let surface = surface::Surface::new(&lib, &wnd).expect("Failed to create surface");

    let hw_list = hw::Description::poll(&lib, Some(&surface)).expect("Failed to list hardware");

    let (hw_dev, queue, _) = hw_list
        .find_first(
            hw::HWDevice::is_discrete_gpu,
            |q| q.is_graphics() && q.is_surface_supported(),
            |_| true
        )
        .expect("Failed to find suitable hardware device");

    let dev_type = dev::DeviceCfg {
        lib: &lib,
        hw: hw_dev,
        extensions: &[extensions::SWAPCHAIN_EXT_NAME],
        allocator: None,
    };

    let device = dev::Device::new(&dev_type).expect("Failed to create device");

    let capabilities = surface::Capabilities::get(&hw_dev, &surface).expect("Failed to get capabilities");

    assert!(capabilities.is_mode_supported(swapchain::PresentMode::FIFO));
    assert!(capabilities.is_flags_supported(memory::UsageFlags::COLOR_ATTACHMENT));

    let surf_format = capabilities.formats().next().expect("No available formats").format;

    let swp_type = swapchain::SwapchainCfg {
        num_of_images: capabilities.min_img_count(),
        format: surf_format,
        color: memory::ColorSpace::SRGB_NONLINEAR,
        present_mode: swapchain::PresentMode::FIFO,
        flags: memory::UsageFlags::COLOR_ATTACHMENT,
        extent: capabilities.extent2d(),
        transform: capabilities.pre_transformation(),
        alpha: capabilities.first_alpha_composition().expect("No alpha composition")
    };

    let swapchain = swapchain::Swapchain::new(&lib, &device, &surface, &swp_type).expect("Failed to create swapchain");

    let vert_shader_type = shader::ShaderCfg {
        path: "VERT_DATA",
        entry: "main",
    };

    let vert_shader =
        shader::Shader::from_glsl(&device, &vert_shader_type, VERT_SHADER, shader::Kind::Vertex)
        .expect("Failed to create vertex shader module");

    let frag_shader_type = shader::ShaderCfg {
        path: "FRAG_DATA",
        entry: "main",
    };

    let frag_shader =
        shader::Shader::from_glsl(&device, &frag_shader_type, FRAG_SHADER, shader::Kind::Fragment)
        .expect("Failed to create fragment shader module");

    let mem_cfg = memory::MemoryCfg {
        properties: hw::MemoryProperty::HOST_VISIBLE,
        filter: &hw::any,
        buffers: &[
            &memory::BufferCfg {
                size: std::mem::size_of_val(VERTEX_DATA) as u64,
                usage: memory::VERTEX,
                queue_families: &[queue.index()],
                simultaneous_access: false,
                count: 1
            },
            &memory::BufferCfg {
                size: std::mem::size_of_val(INDICES) as u64,
                usage: memory::INDEX,
                queue_families: &[queue.index()],
                simultaneous_access: false,
                count: 1
            },
            &memory::BufferCfg {
                size: std::mem::size_of_val(&transformations) as u64,
                usage: memory::UNIFORM,
                queue_families: &[queue.index()],
                simultaneous_access: false,
                count: 1
            },
            &memory::BufferCfg {
                size: std::mem::size_of_val(COLOR_DATA) as u64,
                usage: memory::UNIFORM,
                queue_families: &[queue.index()],
                simultaneous_access: false,
                count: 1
            }
        ]
    };

    let data = memory::Memory::allocate(&device, &mem_cfg).expect("Failed to allocate memory");

    data.access(&mut |bytes: &mut [f32]| {
        bytes.clone_from_slice(VERTEX_DATA);
    }, 0)
    .expect("Failed to fill the buffer");

    data.access(&mut |bytes: &mut [u32]| {
        bytes.clone_from_slice(INDICES);
    }, 1)
    .expect("Failed to fill indices");

    data.access(&mut |bytes: &mut [f32]| {
        bytes.clone_from_slice(&transformations);
    }, 2)
    .expect("Failed to fill coordinate transformations");

    data.access(&mut |bytes: &mut [f32]| {
        bytes.clone_from_slice(COLOR_DATA);
    }, 3)
    .expect("Failed to fill color data");

    let transformations_data = data.resource(2, graphics::ResourceType::UNIFORM_BUFFER, graphics::ShaderStage::VERTEX);
    let color_data = data.resource(3, graphics::ResourceType::UNIFORM_BUFFER, graphics::ShaderStage::FRAGMENT);
    let set = [&transformations_data, &color_data];

    let depth_type = memory::ImageCfg {
        queue_families: &[queue.index()],
        format: memory::ImageFormat::D32_SFLOAT,
        extent: capabilities.extent3d(1),
        usage: memory::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        layout: graphics::ImageLayout::UNDEFINED,
        aspect: memory::ImageAspect::DEPTH,
        properties: hw::MemoryProperty::DEVICE_LOCAL,
    };

    let depth_buffer = memory::Image::new(&device, &depth_type).expect("Failed to allocate depth buffer");

    let render_pass = graphics::RenderPass::with_depth_buffers(&device, surf_format, memory::ImageFormat::D32_SFLOAT, 1)
        .expect("Failed to create render pass");

    let vertex_cfg = [
        graphics::VertexInputCfg {
            location: 0,
            binding: 0,
            format: memory::ImageFormat::R32G32B32A32_SFLOAT,
            offset: 0,
        }
    ];

    let pipe_type = graphics::PipelineCfg {
        vertex_shader: &vert_shader,
        vertex_size: std::mem::size_of::<[f32; 4]>() as u32,
        vert_input: &vertex_cfg,
        frag_shader: &frag_shader,
        topology: graphics::Topology::TRIANGLE_LIST,
        extent: capabilities.extent2d(),
        push_constant_size: 0,
        render_pass: &render_pass,
        subpass_index: 0,
        enable_depth_test: true,
        enable_primitive_restart: false,
        cull_mode: graphics::CullMode::BACK,
        sets: &[&set],
    };

    let pipeline = graphics::Pipeline::new(&device, &pipe_type).expect("Failed to create pipeline");

    pipeline.update(&[&set]);

    let img_sem = sync::Semaphore::new(&device).expect("Failed to create semaphore");
    let render_sem = sync::Semaphore::new(&device).expect("Failed to create semaphore");

    let cmd_pool_type = cmd::PoolCfg {
        queue_index: queue.index(),
    };

    let cmd_pool = cmd::Pool::new(&device, &cmd_pool_type).expect("Failed to allocate command pool");

    let images = swapchain.images().expect("Failed to get images");

    let frames: Vec<memory::Framebuffer> = images.iter()
        .map(|image| {
            let frames_cfg = memory::FramebufferCfg {
                render_pass: &render_pass,
                images: &[&image, &depth_buffer],
                extent: capabilities.extent2d(),
            };

            memory::Framebuffer::new(&device, &frames_cfg).expect("Failed to create framebuffers")
        })
        .collect();

    let cmd_buffers: Vec<cmd::ExecutableBuffer> = frames.iter()
        .map(|frame| {
            let cmd_buffer = cmd_pool.allocate().expect("Failed to allocate command pool");

            cmd_buffer.begin_render_pass(&render_pass, &frame);
            cmd_buffer.bind_graphics_pipeline(&pipeline);
            cmd_buffer.bind_vertex_buffers(&[data.vertex_view(0, vertex_cfg[0].offset)]);
            cmd_buffer.bind_index_buffer(data.view(1), 0, memory::IndexBufferType::UINT32);
            cmd_buffer.bind_resources(&pipeline, &[]);
            cmd_buffer.draw_indexed(INDICES.len() as u32, 1, 0, 0, 0);
            cmd_buffer.end_render_pass();

            cmd_buffer.commit().expect("Failed to commit buffer")
        })
        .collect();

    let queue_cfg = queue::QueueCfg {
        family_index: queue.index(),
        queue_index: 0
    };

    let cmd_queue = queue::Queue::new(&device, &queue_cfg);

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        match event {
            winit::event::Event::WindowEvent {
                event: winit::event::WindowEvent::CloseRequested,
                ..
            } => {
                control_flow.set_exit();
            },
            winit::event::Event::MainEventsCleared => {
                wnd.request_redraw();
            },
            winit::event::Event::RedrawRequested(_) => {
                z_angle += 0.01;

                transformations[48] = z_angle.cos();
                transformations[49] = z_angle.sin();
                transformations[52] = -z_angle.sin();
                transformations[53] = z_angle.cos();

                data.access(&mut |bytes: &mut [f32]| {
                    bytes.clone_from_slice(&transformations);
                }, 2)
                .expect("Failed to fill coordinate transformations");

                let img_index = swapchain.next_image(u64::MAX, Some(&img_sem), None).expect("Failed to get image index");

                let exec_info = queue::ExecInfo {
                    buffer: &cmd_buffers[img_index as usize],
                    wait_stage: cmd::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    timeout: u64::MAX,
                    wait: &[&img_sem],
                    signal: &[&render_sem],
                };

                cmd_queue.exec(&exec_info).expect("Failed to execute queue");

                let present_info = queue::PresentInfo {
                    swapchain: &swapchain,
                    image_index: img_index,
                    wait: &[&render_sem]
                };

                cmd_queue.present(&present_info).expect("Failed to present frame");

                std::thread::sleep(std::time::Duration::from_millis(10));
            },
            _ => ()
        }

    });
}