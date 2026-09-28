from setuptools import find_namespace_packages, setup


setup(
    name="targo",
    version="1.0.0",
    description="TARGO: Benchmarking Target-driven Object Grasping under Occlusions (TARGO-Net training and evaluation).",
    url="https://github.com/TARGO-benchmark/TARGO",
    license="MIT",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_namespace_packages("src", exclude=["*.build", "*.build.*"]),
    py_modules=["utils_giga", "utils_targo"],
    include_package_data=True,
)
