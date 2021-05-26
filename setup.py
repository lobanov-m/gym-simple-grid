import os
import setuptools


def parse_requirements():
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, 'requirements.txt')) as f:
        lines = f.readlines()
    lines = [l for l in map(lambda l: l.strip(), lines) if l != '' and l[0] != '#']
    return lines


requirements = parse_requirements()


setuptools.setup(
    name='arcturus',
    version='1.0',
    # scripts=['arcturus'],
    author="Lobanov Mikhail",
    author_email="m.lobanov@cognitivepilot.com",
    description="Utilities package",
    url="https://gitlab.cognitivepilot.com/ml/utilities/arcturus",
    packages=setuptools.find_packages(),
    classifiers=[
        # "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    install_requires=requirements,
)