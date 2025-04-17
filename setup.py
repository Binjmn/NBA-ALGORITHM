from setuptools import setup, find_packages

setup(
    name="nba_algorithm",
    version="1.0.0",
    description="NBA betting prediction system using machine learning models",
    author="Cascade",
    author_email="info@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "requests",
        "joblib",
        "matplotlib",
        "seaborn",
    ],
    entry_points={
        'console_scripts': [
            'nba-predict=nba_algorithm.scripts.main:main',
            'nba-test=nba_algorithm.scripts.test_predictions:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
