#!/usr/bin/env python3

import os
import sys
import platform
import subprocess
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension
import pkgconfig

# Minimum versions
MINIMUM_PYBIND11_VERSION = "2.10.0"

def check_dependencies():
    """Check for required system dependencies"""
    
    # Check for libsndfile using pkg-config
    try:
        if not pkgconfig.exists('sndfile'):
            raise RuntimeError("libsndfile not found. Please install libsndfile development package.")
    except:
        print("Warning: pkg-config not available or libsndfile not found")
        print("Please ensure libsndfile is installed:")
        print("  Ubuntu/Debian: sudo apt-get install libsndfile1-dev pkg-config")
        print("  macOS: brew install libsndfile pkg-config") 
        print("  CentOS/RHEL: sudo yum install libsndfile-devel pkgconfig")
        raise

def get_atracdenc_sources():
    """Get all atracdenc source files, separated by language"""
    atracdenc_src = Path("atracdenc/src")
    
    if not atracdenc_src.exists():
        raise RuntimeError(
            "atracdenc submodule not found. Please run:\n"
            "git submodule update --init --recursive"
        )
    
    # C++ source files
    cpp_sources = []
    cpp_patterns = [
        "atrac/*.cpp",
        "qmf/*.cpp", 
        "lib/bitstream/*.cpp",
        "lib/mdct/*.cpp",
    ]
    
    for pattern in cpp_patterns:
        files = list(atracdenc_src.glob(pattern))
        # Filter out unit test files
        files = [f for f in files if not f.name.endswith('_ut.cpp')]
        cpp_sources.extend([str(f) for f in files])
    
    # Core source files
    core_files = [
        "aea.cpp", "atrac1denc.cpp", "env.cpp", "help.cpp",
        "pcm_io_sndfile.cpp", "transient_detector.cpp", "wav.cpp"
    ]
    cpp_sources.extend([str(atracdenc_src / f) for f in core_files])
    
    # C source files
    c_sources = [
        str(atracdenc_src / "lib/fft/kissfft_impl/kiss_fft.c"),
        str(atracdenc_src / "lib/liboma/src/liboma.c"),
    ]
    
    return cpp_sources, c_sources

def get_include_dirs():
    """Get include directories"""
    atracdenc_src = Path("atracdenc/src")
    
    includes = [
        str(atracdenc_src),
        str(atracdenc_src / "atrac"),
        str(atracdenc_src / "qmf"), 
        str(atracdenc_src / "lib"),
        str(atracdenc_src / "lib/bitstream"),
        str(atracdenc_src / "lib/fft"),
        str(atracdenc_src / "lib/fft/kissfft_impl"),
        str(atracdenc_src / "lib/mdct"),
        str(atracdenc_src / "lib/liboma/include"),
    ]
    
    return includes

def get_libraries_and_flags():
    """Get library linking information"""
    libraries = []
    library_dirs = []
    extra_compile_args = []
    extra_link_args = []
    include_dirs = []
    define_macros = []
    
    # Get libsndfile info from pkg-config
    try:
        sndfile_cflags = pkgconfig.cflags('sndfile').split()
        sndfile_libs = pkgconfig.libs('sndfile').split()
        
        # Parse cflags for include directories
        for flag in sndfile_cflags:
            if flag.startswith('-I'):
                include_dirs.append(flag[2:])
            else:
                extra_compile_args.append(flag)
        
        # Parse libs for libraries and library directories
        for flag in sndfile_libs:
            if flag.startswith('-L'):
                library_dirs.append(flag[2:])
            elif flag.startswith('-l'):
                libraries.append(flag[2:])
            else:
                extra_link_args.append(flag)
                
    except Exception as e:
        print(f"Warning: Could not get libsndfile info from pkg-config: {e}")
        # Fallback
        libraries.append('sndfile')
        
    # Initialize define_macros list
    define_macros = []
    
    # Platform-specific
    if platform.system() == 'Linux':
        libraries.extend(['m'])  # Math library
        extra_compile_args.extend(['-fopenmp', '-O3', '-DNDEBUG', '-std=c++17'])
        extra_link_args.extend(['-fopenmp'])
        define_macros.append(('_OPENMP', '1'))
    elif platform.system() == 'Darwin':
        # macOS - use clang's OpenMP
        extra_compile_args.extend(['-Xpreprocessor', '-fopenmp', '-O3', '-DNDEBUG', '-std=c++17'])
        extra_link_args.extend(['-lomp'])
        define_macros.append(('_OPENMP', '1'))
    elif platform.system() == 'Windows':
        extra_compile_args.extend(['/openmp', '/O2', '/DNDEBUG', '/std:c++17'])
        define_macros.append(('_OPENMP', '1'))
        
    return libraries, library_dirs, extra_compile_args, extra_link_args, include_dirs, define_macros


class CustomBuildExt(build_ext):
    """Custom build extension to handle C files separately."""
    
    def build_extension(self, ext):
        if ext.name == "pytrac":
            self.build_c_objects(ext)
        super().build_extension(ext)
    
    def build_c_objects(self, ext):
        """Compile C sources separately with C compiler"""
        c_sources = []
        cpp_sources = []
        
        # Separate C and C++ sources
        for source in ext.sources:
            if source.endswith('.c'):
                c_sources.append(source)
            else:
                cpp_sources.append(source)
        
        # Compile C sources with C compiler
        if c_sources:
            # Get C compiler
            c_compiler = self.compiler.compiler_so[0]  # Usually 'clang' or 'gcc'
            
            c_objects = []
            for c_source in c_sources:
                # Create object file path
                obj_file = self.compiler.object_filenames([c_source], strip_dir=0)[0]
                obj_file = os.path.join(self.build_temp, obj_file)
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(obj_file), exist_ok=True)
                
                # Compile with C compiler and C flags
                c_cmd = [c_compiler, '-c', c_source, '-o', obj_file, '-std=c99']
                c_cmd.extend([f'-I{inc}' for inc in ext.include_dirs])
                c_cmd.extend([f'-D{name}={value}' for name, value in ext.define_macros])
                c_cmd.extend(['-fPIC'])  # Position independent code
                
                print(f"Compiling C file: {' '.join(c_cmd)}")
                subprocess.run(c_cmd, check=True)
                c_objects.append(obj_file)
            
            # Add compiled C objects to extra objects
            if not hasattr(ext, 'extra_objects'):
                ext.extra_objects = []
            ext.extra_objects.extend(c_objects)
        
        # Update sources to only include C++ sources
        ext.sources = cpp_sources


# Check dependencies
check_dependencies()

# Get source files and includes
cpp_sources, c_sources = get_atracdenc_sources()
include_dirs = get_include_dirs()
libraries, library_dirs, extra_compile_args, extra_link_args, sndfile_includes, define_macros = get_libraries_and_flags()

# Add sndfile include directories
include_dirs.extend(sndfile_includes)

# Compiler definitions
define_macros = [
    ('VERSION_INFO', '"dev"'),
    ('kiss_fft_scalar', 'float'),
    ('PYBIND11_DETAILED_ERROR_MESSAGES', '1'),
]

# Create main pybind11 extension with all sources
pytrac_ext = Pybind11Extension(
    "pytrac",
    sources=["pytrac_bindings.cpp"] + cpp_sources + c_sources,
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    define_macros=define_macros,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    cxx_std=17,
)

ext_modules = [pytrac_ext]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False
)
