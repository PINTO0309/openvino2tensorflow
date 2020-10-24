from setuptools import setup, Extension
from setuptools import find_packages

with open("README.md") as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name="openvino2tensorflow",
        scripts=["scripts/openvino2tensorflow"],
        version="0.0.1",
        description="This script converts the OpenVINO IR model to Tensorflow's saved_model, tflite, h5 and pb. in (NCHW) format",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Katsuya Hyodo",
        author_email="rmsdh122@yahoo.co.jp",
        url="https://github.com/PINTO0309/openvino2tensorflow",
        license="MIT License",
        packages=find_packages(),
        platforms=["linux", "unix"],
        python_requires=">3.6",
    )