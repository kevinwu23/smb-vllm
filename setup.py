from setuptools import setup, find_packages

setup(
    name="multimodal-qwen3-vllm",
    version="0.1.0",
    description="Multimodal LLM extension for vLLM supporting arbitrary embeddings",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Multimodal LLM Team",
    author_email="team@example.com",
    url="https://github.com/example/multimodal-qwen3-vllm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "vllm>=0.6.0",
        "numpy>=1.20.0",
        "pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
        ],
    },
) 