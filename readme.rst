ORR Optimization
=================

Catalyst structure optimization for the oxygen reduction reaction on Pt and Au.

Dependencies
-------------
* `Atomic simualtion environment <https://wiki.fysik.dtu.dk/ase/>`_ : Data structures for molecular structures and file IO.

Publications
-------------
* M. Nunez, J. Lansford, D.G. Vlachos, "Optimization of transition metal catalyst facet structure: Application to the oxygen reduction reaction" (under revision)

Developers
-----------
* Marcel Nunez (mpnunez28@gmail.com)
* Joshua Lansford (lansford.jl@gmail.com )

Directory Structure
--------------------
* volcano: produces all_volcanos.npy, which is read by the files in the structures folder.
* structure: Provides classes for simulating the ORR chemistry and catalyst structure
* optimization_scripts: folders run the optimizations and do the analysis by importing classes from the structures folder
* figures: produces figures for publication