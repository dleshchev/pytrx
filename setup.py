from setuptools import setup, find_packages

setup(
    name='pytrx',
    version='0.1.0',
    author='Denis Leshchev',
    author_email='leshchev.denis@gmail.com',
    packages=find_packages("."),
    url='https://github.com/dleshchev/pytrx',
    license='LICENSE.txt',
    description='toolbox for analysis of time-resolved x-ray experiments',
    long_description=open('README.md').read(),
    include_package_data=True,
	install_requires=[
        "numpy",
        "fabio",
		"pandas",
		"matplotlib",
		"scipy",
    ],
)