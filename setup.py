from setuptools import setup, find_packages

setup(
    name='microjax',
    version='0.0.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=["jax","numpy"],
)
