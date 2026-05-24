{
  pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  }
}:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [
      (ps.cupy.overrideAttrs (old: {
        preConfigure =
          let
            cudaArch = builtins.getEnv "CUDA_ARCH";
            arch = "arch=compute_${cudaArch},code=sm_${cudaArch}";
          in ''
            ${old.preConfigure}
            echo "Setting CUPY_NVCC_GENERATE_CODE to ${arch}"
            export CUPY_NVCC_GENERATE_CODE="${arch}"
          '';
      }))
      ps.numpy
      ps.pyqtgraph
      ps.pyqt6
      ps.scipy
      ps.tiktoken
      ps.tqdm
    ]))
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.cuda_nvrtc
    pkgs.cudaPackages.cuda_cudart
    pkgs.cudaPackages.libcublas
  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.cudaPackages.cuda_nvrtc
      pkgs.cudaPackages.cuda_cudart
      pkgs.cudaPackages.libcublas
    ]}:$LD_LIBRARY_PATH
  '';
}
