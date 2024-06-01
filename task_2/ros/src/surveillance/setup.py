from glob import glob

from setuptools import find_packages, setup

package_name = "surveillance"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author=["Daniele Paccusse", "Mattia Guazzaloca"],
    author_email=[
        "daniele.paccusse@studio.unibo.it",
        "mattia.guazzaloca@studio.unibo.it",
    ],
    description="Distributed algorithm for autonomous robot to move forward targets while keeping the team in formation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "warden = surveillance.warden:main",
            "plotter = surveillance.plotter:main",
        ],
    },
)
