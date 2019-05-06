ORR Optimization
=================

Catalyst structure optimization for the oxygen reduction reaction on Pt and Au.

Dependencies
-------------
* `Atomic simualtion environment <https://wiki.fysik.dtu.dk/ase/>`_ : Data structures for molecular structures and file IO.

Publications
-------------
* `M. Núñez, J. Lansford and D.G Vlachos, “Optimization of the facet structure of transition-metal catalysts applied to the oxygen reduction reaction” Nature Chemistry (2019) <https://www.nature.com/articles/s41557-019-0247-4>`_

Developers
-----------
* Marcel Nunez (mpnunez28@gmail.com)
* Joshua Lansford (lansford.jl@gmail.com )

Directory Structure
--------------------
* volcano: produces all_volcanos.npy, which is read by the files in the structures folder.
* orr_: Provides classes for simulating the ORR chemistry and catalyst structure
* optimization_scripts: folders run the optimizations and do the analysis by importing classes from the structures folder
* figures: produces figures for publication

Usage
-------
* Add repository to PYTHONPATH environment variable
* ```import orr_optimizer``` or run scripts in the scripts folder