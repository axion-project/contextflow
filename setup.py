from setuptools import setup, find_packages

setup(
    name="orac",
    version="0.1.0",
    description="ORAC AI Framework: Advanced Transformer with Persistent Memory and Adaptive Modes",
    author="Mayhem",
    author_email="michael@aedininsight.com",
    url="https://aedininsight.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.21.0",
        "faiss-cpu>=1.7.2",
        "sentence-transformers>=2.2.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="transformer AI memory retrieval faiss sentence-transformers deep learning",
)
