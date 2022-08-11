
<!-- <a name="readme-top"></a> -->
<p><a id="readme-top"></a></p>




<br />
<div align="center">
  <a href="https://sites.psu.edu/reinhartgroup/">
    <img src="Sample_figure\drawing\data_driven_logo.png" alt="Logo" width="660" height="156">
  </a>

<h2 align="center">Evaluating generative models for inverse design of high-entropy refractory alloys</h2>

  <p align="center">
    This repository presents my master project achievement which is focus on comparing the effectiveness of using three types of generative models, Conditional Autoencoder, Conditional Variational Autoencoder and Conditional Generative Adversarial Network on the research of high entropy alloys. Welcome to any suggestions!
    <br />
    <a href="https://github.com/Ellisontung/HEAs_project/blob/master/Evaluating%20generative%20models%20for%20inverse%20design%20of%20high-entropy%20refractory%20alloys.pdf"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Ellisontung/HEAs_project/blob/master/generative_models_comparison.ipynb">View Demo</a>
    ·
    <a href="https://github.com/Ellisontung/HEAs_project/issues">Report Bug</a>
    ·
    <a href="https://github.com/Ellisontung/HEAs_project/issues">Request Feature</a>
  </p>
</div>




<details>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#Dataset components">Dataset components</a>
      <ul>
        <li><a href="#Elements">Elements</a></li>
        <li><a href="#Properties">Properties</a></li>
      </ul>
    </li>
    <li>
        <a href="#Models Architectures">Models Architectures</a>
        <ul>
            <li><a href="#cAE">Conditional Autoencoder</a></li>
            <li><a href="#cVAE">Conditional Variational Autoencoder</a></li>
            <li><a href="#cGAN">Conditional Generative Adversarial Network</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>




## About The Project

Generative modeling is an innovative new method to design functional materials. In this project, we quantify the performance of several generative modeling architectures – autoencoder, variational autoencoder, and generative adversarial network – to design novel materials. These are compared to rational design method by case study of refractory high-entropy alloys for ultra-high-temperature applications. Furthermore, we apply a series of validation methods to evaluate the models' effectiveness and express the current difficulty. Overall, cAE is able to create versatile compositions and keep at lower similarity; cVAE is capable of generating compositions with high accuracy; WcGAN has ability of creating novel compositions compared to training data. However, since the mode collapse phenomenon occurred in our results, inverse design still cannot totally replace rational design method at this stage.

<div align="center">
  <a href="https://github.com/Ellisontung/HEAs_project/blob/master/Sample_figure/drawing/Inverse%20design.png">
    <img src="Sample_figure\drawing\Inverse design.png" alt="Logo" width="300" height="300">
  </a>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<p><a id="Dataset components"></a></p>

## Dataset components

The dataset used in the project contains 529 different high entropy alloys. Each composition contains a recipe (percentage of elements) and the properties of shear modulus and fracture toughness itself.
<div align=center id="#Dataset components">
    <a href="https://github.com/Ellisontung/HEAs_project/blob/master/srf/SM_FT_dataset.csv"> Source file</a>

</div>

<p><a id="Elements"></a></p>

### Elements

In the figure below, it demos five of compositions. Darker blue represents higer percentage of the element (white indicates zero percentage).
<div align="center">
  <a href="https://github.com/Ellisontung/HEAs_project/blob/master/Sample_figure/dataset_samples.pdf">
    <img src="Sample_figure\dataset_samples.png" alt="Logo" width="720" height="240">
  </a>
</div>

<p><a id="Properties"></a></p>

### Properties

In the figure below, it demos plots all the datapoints in scatter plot (x-axis represents toughness value; y-axis represents shear modulus value).
<div align="center">
  <a href="https://github.com/Ellisontung/HEAs_project/blob/master/Sample_figure/scatter_plot.pdf">
    <img src="Sample_figure\Training_data.png" alt="Logo" width="720" height="480">
  </a>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<p><a id="Models Architectures"></a></p>

## Models Architectures


<p><a id="cAE"></a></p>

### Conditional Autoencoder
<div align="center">
  <a href="https://github.com/Ellisontung/HEAs_project/blob/master/Sample_figure/drawing/AE.png">
    <img src="Sample_figure\drawing\AE.png" alt="Logo" width="720" height="480">
  </a>
</div>


<p><a id="cVAE"></a></p>

### Conditional Variational Autoencoder
<div align="center">
  <a href="https://github.com/Ellisontung/HEAs_project/blob/master/Sample_figure/drawing/cVAE.png">
    <img src="Sample_figure\drawing\cVAE.png" alt="Logo" width="720" height="480">
  </a>
</div>


<p><a id="cGAN"></a></p>

### Conditional Generative Adversarial Network
<div align="center">
  <a href="https://github.com/Ellisontung/HEAs_project/blob/master/Sample_figure/drawing/cGAN.png">
    <img src="Sample_figure\drawing\cGAN.png" alt="Logo" width="720" height="480">
  </a>
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

To use the trained model, the three trained models have been uploaded under the folder, ["Trained model"](https://github.com/Ellisontung/HEAs_project/blob/master/generative_models_comparison.ipynb), and prepared to be used, please see the example in the file gnerative_models_compairson.
To train the new model, can follow the steps which have been setup in the notebooks ["train_cAE"](https://github.com/Ellisontung/HEAs_project/blob/master/train_AE.ipynb) and ["train_GAN"](https://github.com/Ellisontung/HEAs_project/blob/master/train_WcGAN.ipynb).
<br></br>
To view full comparison results and details please read the [Paper](https://github.com/Ellisontung/HEAs_project/blob/master/Evaluating%20generative%20models%20for%20inverse%20design%20of%20high-entropy%20refractory%20alloys.pdf).

<p align="right">(<a href="#readme-top">back to top</a>)</p>








## Contact

Name: Yen Cheng Tung (Ellison)
<br></br>
Email: tung861103@gmail.com
<br></br>
LinkedIn: https://www.linkedin.com/in/yenchengtung/

<p align="right">(<a href="#readme-top">back to top</a>)</p>




## Acknowledgments

* Dr. Reinhart group, Pennsylvania State University
* Department of Material Science, Pennsylvania University

<p align="right">(<a href="#readme-top">back to top</a>)</p>






[linkedin-url]: https://www.linkedin.com/in/yenchengtung/
[product-screenshot]: https://github.com/Ellisontung/HEAs_project/blob/master/Sample_figure/drawing/Inverse%20design.png

[scikit-learn-url]: https://scikit-learn.org/stable/
[pytorch-url]: https://pytorch.org/
[pytorch.com]: https://img.shields.io/badge/PyTorch-PyTorch-orange
[matplotlib-url]: https://matplotlib.org/
[panadas-url]: https://pandas.pydata.org/
[numpy-url]: https://pandas.pydata.org/
[pymatgen-url]: https://pymatgen.org/