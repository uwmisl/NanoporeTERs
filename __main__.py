#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p "python3.withPackages(ps: [ ps.numpy  ps.jupyter ])"


print("Importing argparse")
import argparse
print("Argparse imported")

print("Importing numpy")
import numpy 
print("Numpy imported")

print("Importing juypter")
import jupyter
print("Argpajupyterrse imported")

purpose = "Friendship"

#purpose = input("All this...And for what?")
print(f"HA! {purpose}? {purpose.upper()}?? You wouldn't know '{purpose}' if it hit you in the face!")