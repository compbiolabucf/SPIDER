from setuptools import Command, find_packages, setup

__lib_name__ = "SPIDER"
__lib_version__ = "1.0.1"
__description__ = "SPIDER: Spatially Integrated Denoising via Embedding Regularization with Single Cell Supervision"
__url__ = "https://github.com/compbiolabucf/SPIDER"
__author__ = "MD Istiaq Ansari"
__author_email__ = "istiaq@ucf.edu"
__license__ = "MIT"
__keywords__ = ["spatial transcriptomics", "Deep learning", "Graph attention auto-encoder"]
__requires__ = ["requests",]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['SPIDER'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
)