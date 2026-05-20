from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME    = "fraud-detection-ml"
AUTHOR_USER_NAME = "korede-folarin"
SRC_REPO     = "frauddetection"
AUTHOR_EMAIL = "your-email@email.com"

setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Banking ML Risk Pipeline — Fraud Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src")
)