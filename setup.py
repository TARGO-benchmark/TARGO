from setuptools import find_namespace_packages, setup


setup(
    name="targo",
    version="0.1.0",
    description="Target-aware robotic grasping models and evaluation utilities.",
    package_dir={"": "src"},
    packages=find_namespace_packages("src"),
    py_modules=["utils_giga", "utils_targo", "csv_creation"],
    include_package_data=True,
)
