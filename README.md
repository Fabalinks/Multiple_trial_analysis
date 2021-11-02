<div id="top"></div>
<!--

-->



<!-- PROJECT SHIELDS -->
<!--

-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Fabalinks/Multiple_trial_analysis">
    <img src="images/logo.png" alt="Logo" width="160" height="160">
  </a>

<h3 align="center">Rearing analysis</h3>

  <p align="center">
    Dedicated to analyzing rearign behavior in rats in Ratcave VR
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#data-description">Data Description</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<br />
<div align="center">
  <a href="https://github.com/Fabalinks/Multiple_trial_analysis">
    <img src="images/Wireless-movie0001-0249-demo3 (1).gif" alt="Wireless recording" width="550" height="400">
  </a>
</div>


Research to uncover the neural basis behind Path integration. 
In this project we use virtual reality with freely moving rodents while recording
electrophysiological signal from the Hippocampus CA1.
Animals are taught to rear at a visible beacon in the virtual arena, 
retrieve a randomly distributed reward and go back to the original location
where the beacon was, but this time in darkness. We focus on understanding
how the animal accumulates vectors as it travels to the reward and then
when it needs to retrieve/calculate a correct vector to the original location. 
In this analysis we focus on proving the rearing behavior at the beacon. 
 


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
### Installation

 * pip
  ```sh
  pip install https://github.com/Fabalinks/Multiple_trial_analysis
  ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- Data Description -->
## Data description

During an experimental session 3 files are generated. 

position datetime.txt (rotation in quaternion coordinates):

| Time     | X rat | Y rat | Z rat | X rotation_rat | Y rotation_rat| Z rotation_rat | Motive Frame|Motive timestamp|Motive session timestamp|
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 


beacons datetime.txt:

 | Time     | X rat | Y rat | Z rat | X beacon | Y beacon| 
 | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | 

metadata datetime.txt

self explanatory text file that can be machine read out




<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b analysis/New_stuff`)
3. Commit your Changes (`git commit -m 'Add some New_stuff'`)
4. Push to the Branch (`git push origin analysis/New_stuff`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the Apache License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Fabian Stocek - [@fabalinks](https://twitter.com/@fabalinks) - stocek@bio.lmu.de

Project Link: [https://github.com/Fabalinks/Multiple_trial_analysis](https://github.com/Fabalinks/Multiple_trial_analysis)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Sirota lab](https://cogneuro.bio.lmu.de/people/group-members/sirota/index.html)
* [ Funding - RTG 2175](https://www.rtg2175.bio.lmu.de/index.html)


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Fabalinks/Multiple_trial_analysis.svg?style=for-the-badge
[contributors-url]: https://github.com/Fabalinks/Multiple_trial_analysis/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Fabalinks/Multiple_trial_analysis.svg?style=for-the-badge
[forks-url]: https://github.com/Fabalinks/Multiple_trial_analysis/network/members
[stars-shield]: https://img.shields.io/github/stars/Fabalinks/Multiple_trial_analysis.svg?style=for-the-badge
[stars-url]: https://github.com/Fabalinks/Multiple_trial_analysis/stargazers
[issues-shield]: https://img.shields.io/github/issues/Fabalinks/Multiple_trial_analysis.svg?style=for-the-badge
[issues-url]: https://github.com/Fabalinks/Multiple_trial_analysis/issues
[license-shield]: https://img.shields.io/github/license/Fabalinks/Multiple_trial_analysis.svg?style=for-the-badge
[license-url]: https://github.com/Fabalinks/Multiple_trial_analysis/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/fabian-stocek/
[product-screenshot]: images/screenshot.png
