from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("mmrag/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mmrag",
    version="0.1.0",
    author="Abhijoy Sarkar",
    description="Multimodal Retrieval-Augmented Generation system combining vision and text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhijoysarkar/mmrag",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.4,<2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mmrag=mmrag.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mmrag": ["configs/*.yaml"],
    },
)
