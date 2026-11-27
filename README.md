<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>

<div align="center">
  <h3>KESTREL</h3>

  <p>
    A ligand-field program for 3d<sup>n</sup> transition-metal complexes
    <br />
    <a href="https://github.com/chem-kestrel/kestrelpy/issues/new?labels=bug&amp;template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/chem-kestrel/kestrelpy/issues/new?labels=enhancement&amp;template=feature-request---.md">Request Feature</a>
    &middot;
    <a href="https://chem-kestrel.github.io">Website</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#features">Features</a></li>
        <li><a href="#citing-this-project">Citing This Project</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

KESTREL is a free-to-use ligand-field program, which uses metal-ligand bonding parameters to simulate the ligand-field around the central metal ion.

### Features

* Multiconfigurational calculations.
* Orbital projections of ground and excited states.
* Handle any d(n) configuration, n=1..9.
* Easily include spin-orbit coupling and interelectronic repulsion.
* Simulate UV-Vis, CD, and MCD spectra for non-centrosymmetric complexes.
* Calculate EPR properties (g-factors and metal-hyperfine).
* Calculate paramagnetic susceptibilities.
* Calculate single-ion molecular magnetic properties.

### Citing This Project

An official release paper is still being worked on. In the meantime, please cite this repository or the following works based on an older developmental version of the software:
1.  Nunn, George (2023) Development of a contemporary ligand-field
    program for 3d transition-metal complexes. PhD thesis,
    University of York.
2.  P. Comba, G. Nunn, F. Scherz and P. H. Walton, Faraday Discuss., 2021, 234, 232–244.
3.  S. Mace, B. S. Grace, A. Goodfellow, M. A. Bates, M. A. Halcrow, P. H .Walton, Y. Xu and B. N. Nguyen, Inorg. Chem. Front., 2026, 13 (17), 7301-7315

### Built With

* [Numpy](https://github.com/numpy/numpy)
* [Scipy](https://github.com/scipy/scipy)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

The project uses [poetry](https://python-poetry.org/) for dependency management.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/chem-kestrel/kestrelpy.git
   ```
2. Install packages
   ```sh
   poetry install
   ```
3. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```
4. Check everything works by running
   ```sh
   poetry run pytest
   ```
5. *Optional*: install the pre-commit hooks
   ```sh
   poetry run pre-commit install
   ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Please see the [examples](https://github.com/chem-kestrel/kestrelpy/tree/main/examples) folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/chem-kestrel/kestrelpy/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=chem-kestrel/kestrelpy" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the MIT license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

For support or general enquiries, please reach out to: chem-kestrel@york.ac.uk

Project Link: [https://github.com/chem-kestrel/kestrelpy](https://github.com/chem-kestrel/kestrelpy)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

Copyright (c) 2026 <br> - George Nunn, Benjamin S. Grace, Martin A. Bates and Paul H. Walton
<br> - Department of Chemistry, University of York (UK)
