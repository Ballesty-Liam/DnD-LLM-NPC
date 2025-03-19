from setuptools import setup, find_packages

setup(
    name="radiant_citadel_npc",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "langchain>=0.1.0",
        "sentence-transformers",
        "transformers",
        "torch",
        "accelerate",
        "faiss-cpu",
        "pypdf",
        "markdown",
        "python-dotenv",
        "typer",
        "rich",
        "fastapi",
        "uvicorn",
        "jinja2",
        "tqdm",
        "requests",
        "beautifulsoup4"
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "thallan=app.cli:app",
            "thallan-web=app.web:run_server",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI-powered D&D NPC from the Radiant Citadel",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/radiant_citadel_npc",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)