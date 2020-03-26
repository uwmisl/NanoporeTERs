let
  nixpkgs = import <nixpkgs> {};
  packages = nixpkgs.python37.pkgs;

  run_pkgs = (import ./nix/runtime_dependencies.nix);
  build_pkgs = (import ./nix/build_dependencies.nix);

  # Base python package definition, which we'll then 
  nanoporeTERs = packages.buildPythonApplication {
    pname = "nanoporterpy";
    version = "0.0.1";

    src = ./.;

    doCheck = true;
    doInstallCheck = true;

    # Build-time exclusive dependencies
    buildInputs = [ ];
 
    # Test Dependencies 
    checkInputs = [
      packages.green # Test-runner 
      packages.black # Opinionated code formatter
    ];

    # Run-time dependencies
    propagatedBuildInputs = [
        packages.numpy
        packages.pandas
        packages.h5py
        packages.dask
        packages.matplotlib
        packages.seaborn
        packages.notebook
        packages.jupyter
    ];
    
    installPhase = ''
        mkdir -p $out/bin
        cp ${./__main__.py} $out/bin/nanoporeter
    '';

  };

in nanoporeTERs