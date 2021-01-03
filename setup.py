import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="helix",
    version="0.1.1",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="Helix parameterization package [alpha version]",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/colorsimple",
    keywords = ['helix', 'parameterization'],
    packages=setuptools.find_packages(),
    install_requires = ['numpy', 'lmfit'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
