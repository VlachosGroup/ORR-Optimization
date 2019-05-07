ORR Optimization
=================

Catalyst structure optimization for the oxygen reduction reaction on Pt and Au.

Dependencies
-------------
* `Atomic simualtion environment (3.17.0) <https://wiki.fysik.dtu.dk/ase/>`_ : Data structures for molecular structures and file IO.
* See requirements.txt for suitable virtual environment dependencies

Publications
-------------
* `M. Núñez, J. Lansford and D.G Vlachos, “Optimization of the facet structure of transition-metal catalysts applied to the oxygen reduction reaction” Nature Chemistry (2019) <https://www.nature.com/articles/s41557-019-0247-4>`_

Developers
-----------
* Marcel Nunez (mpnunez28@gmail.com)
* Joshua Lansford (lansford.jl@gmail.com )

Directory Structure
--------------------
* orr_optimizer: Core classes for simulating the ORR chemistry and catalyst structure
* volcano: generate_volcanoes.py produces all_volcanos.npy, which is read by the files in the structures folder.
* optimization_scripts: Optimize_main.py performs a sample optimization. Extract_data.py shows data from the publication.
* figures: produces some figures from publication

Usage
-------
* Add repository to PYTHONPATH environment variable
* ```import orr_optimizer``` or run scripts in the scripts folder
