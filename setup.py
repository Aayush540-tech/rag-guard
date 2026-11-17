from setuptools import setup, find_packages

setup(
    name="rag-guard",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"rag_guard": ["weights.json"]},
    include_package_data=True,
)
