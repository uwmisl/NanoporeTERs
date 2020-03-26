/*
* shell.nix 
* 
* This Nix expression provides a shell environment developers to test 
* This is most useful for ongoing development, testing new dependencies, etc.
*/

with import <nixpkgs> {};

let run_dependencies = (import ./runtime_dependencies.nix);
    build_dependencies = (import ./build_dependencies.nix);
# For local usage
#let pythonEnv = python37.withPackages (python37Packages: [

    pythonEnv = python37.withPackages (python37Packages:
        run_dependencies python37Packages
     ++ build_dependencies python37Packages
    /*
    [
    #python37.withPackages (python37Packages: [
         

        python37Packages.numpy
        python37Packages.pandas
        python37Packages.h5py
        python37Packages.dask
        python37Packages.matplotlib
        python37Packages.seaborn
        python37Packages.notebook

        # For interactive builds
        python37Packages.jupyter
    ]
    */
    ); 

in mkShell {
    buildInputs = [
        pythonEnv
    ];
}

