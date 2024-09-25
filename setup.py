from setuptools import setup, find_packages


setup(
    name="qalo",  # Replace with your package name
    version="1.0.0",  # Initial version number
    author="Zhihao Xu",  # Replace with your name
    author_email="zxu8@nd.edu",  # Replace with your email
    description="quantum annealing assisted lattice optimization",  # Replace with a short description
    packages=find_packages(),  # Automatically find and include all packages in the directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license (if different)
        "Operating System :: OS Independent",
    ]
)
