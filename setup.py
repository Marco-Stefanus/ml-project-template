from setuptools import setup, find_packages

setup(
    name="ml_project_template",
    version="0.1.0",
    description="A modular ML project template.",
    author="Your Name",
    author_email="your@email.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "loguru",
        "pyyaml",
        "joblib"
    ],
    include_package_data=True,
    zip_safe=False,
)
