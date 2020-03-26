# build_dependencies.nix
# 
# This expression hosts the project's build-only dependencies (e.g. packages needed for the code to build 
# the project). This excludes runtime-only dependencies, like numpy.
#
# It takes in a pythonPackage, which is intended to be provided by `python.withPackages`
pythonPackages: [
        # Add new build-only or test-only dependencies below
        
        # Test-runner
        pythonPackages.green
        # Opinionated code-formatter
        pythonPackages.black
        # Docstring static analyzer
        pythonPackages.pydocstyle
]