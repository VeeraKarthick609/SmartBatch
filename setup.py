from setuptools import setup, find_packages

setup(
    name="smartbatch",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "prometheus_client",
        "msgpack",
        "torch",
        "numpy",
    ],
)
