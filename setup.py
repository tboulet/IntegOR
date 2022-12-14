from setuptools import setup, find_namespace_packages

setup(
    name="integor",
    url="https://github.com/tboulet/IntegOR", 
    author="Timothé Boulet",
    author_email="timothe.boulet0@gmail.com",
    
    packages=find_namespace_packages(),

    version="1.1",
    license="MIT",
    description="Optimization and constraint satisfaction for integer linear programs in a user-friendly way",
    long_description=open('README.md').read(),      # always in md, with a README.md (convention!)
    long_description_content_type="text/markdown",  # always in md !
)