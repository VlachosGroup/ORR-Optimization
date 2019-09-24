import setuptools
with open("readme.rst", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='ORR-Optimization',
     version='0.1',
     scripts=[] ,
     author="Marcel Nunez",
     author_email="mpnunez28@gmail.com",
     description="Catalyst structure optimization for the oxygen reduction reaction on Pt and Au",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/VlachosGroup/ORR-Optimization",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
