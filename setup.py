import setuptools

requirements = [
    # use environment.yml
]

setuptools.setup(
    name="TestRLFrameworkPleaseIgnore",
    version="0.1",
    url="https://github.com/JuanPedroGHM/TestRLFrameworkPleaseIgnore",
    author="Juan Pedro Gutierrez",
    author_email="juanpedroghm@gmail.com",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "trlfpi=trlfpi.cli:trlfpi"
        ]
    },
    install_requires=requirements,

)
