# Accelerator device abstraction library

accelerator_sources = files(
    'accelerator_device.cpp',
    'neuron_accelerator_device.cpp',
)

accelerator_headers = files(
    'accelerator_device.h',
    'null_accelerator_device.h',
    'neuron_accelerator_device.h',
)

# Conditional CUDA support
if cuda_dep.found()
    accelerator_sources += files(
        'cuda_accelerator_device.cpp',
    )
    accelerator_headers += files(
        'cuda_accelerator_device.h',
    )
endif

# Create static library
accelerator_lib = static_library(
    'nixl_accelerator',
    accelerator_sources,
    include_directories: [nixl_inc, utils_inc],
    dependencies: [cuda_dep, hwloc_dep],
    install: false,
)

# Dependency for other components
accelerator_dep = declare_dependency(
    link_with: accelerator_lib,
    include_directories: include_directories('.'),
)

# Install headers
install_headers(
    accelerator_headers,
    subdir: 'nixl/accelerator',
)
