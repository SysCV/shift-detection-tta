<div align="center">
  <img src="resources/shift-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <a href="https://www.vis.xyz/shift/">
    <b><font size="5">SHIFT dataset website</font></b>
    </a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://www.vis.xyz/">
    <b><font size="5">VIS Group website</font></b>
    </a>
  </div>
  <div>&nbsp;</div>
</div>

## Introduction

SHIFT is a driving dataset for continuous multi-task domain adaptation. It is a maintained by the [VIS](https://www.vis.xyz/) group at ETH Zurich.

The master branch works with **PyTorch1.6+**.

# TODO: put teaser video for SHIFT detection under domain shift here
<div align="center">
  <img src="https://user-images.githubusercontent.com/24663779/103343312-c724f480-4ac6-11eb-9c22-b56f1902584e.gif" width="800"/>
</div>

## Get Started

Please refer to [get_started.md](docs/get_started.md) for install instructions.

Please refer to [inference.md](docs/user_guides/inference.md) for the basic usage of our repository. If you want to train and test your own model, please see [dataset_prepare.md](docs/user_guides/dataset_prepare.md) and [train_test.md](docs/user_guides/train_test.md).

## Model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

### Object Detection

Supported Methods

- [x] [mean_teacher](configs/sot/mean_teacher)

Supported Datasets

- [x] [SHIFT](https://www.vis.xyz/shift/)


## Citation

If you find this project useful in your research, please consider cite:

```latex
@inproceedings{sun2022shift,
  title={SHIFT: a synthetic driving dataset for continuous multi-task domain adaptation},
  author={Sun, Tao and Segu, Mattia and Postels, Janis and Wang, Yuxuan and Van Gool, Luc and Schiele, Bernt and Tombari, Federico and Yu, Fisher},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21371--21382},
  year={2022}
}
```

## License

This project is released under the [MIT License](LICENSE).
