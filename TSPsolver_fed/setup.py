from setuptools import setup, find_packages

setup(
    name="TSPsolver_fd9",
    version="0.1.0",
    description="A library for solving the Traveling Salesman Problem using various algorithms.",
    author="Frank Dadzie",
    author_email="frank.dadzie@students.cau.edu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
