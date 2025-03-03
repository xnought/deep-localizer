<img src="https://github.com/user-attachments/assets/b00dc021-8e31-44be-9a69-ba33ed8054c6" width="800px">

**DeepLocalizer**: Quickly find functional specialization in deep neural networks. 

Extends [The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units](https://arxiv.org/abs/2411.02280) and [Brain-like functional specialization emerges spontaneously in deep neural networks](https://web.mit.edu/bcs/nklab/media/pdfs/Dobs_2022.pdf) to other models and data.

[Face Localizer Resnet Report](https://www.donnybertucci.com/project/deeplocalizer) (IN PROGRESS) 

> [!NOTE]
> Currently working on the first example (resnet face localizer) in the /experiments folder
>
> Once this is done, I'll write a report + write the library

**Roadmap**

- [x] Replicate some parts of original paper (https://github.com/xnought/paper-implement/tree/main/language_network)
- [ ] Write code from scratch to do analysis on face with resnet
	- [x] Set up face localizer example w/ goal of applying to a resnet model
		- [x] 5k positive (faces) from CelebA
		- [x] 5k negative (objects) from COCO
	- [x] Extract activations from the resnet model
		- [x] test track the activations
		- [x] store activations on disk
	- [x] Contrast positive vs negative activations
	- [ ] Ablation w/ statistical tests on resnet
		- [x] write code to ablate torch models easily 
		- [ ] ablate given the top percent face activations
		- [ ] Compare performance after ablation 
- [ ] Write general API from most helpful functions so others can easily use the library 


## References

**papers**
- https://arxiv.org/abs/2411.02280
- https://web.mit.edu/bcs/nklab/media/pdfs/Dobs_2022.pdf

**code/datasets**
- https://huggingface.co/datasets/tpremoli/CelebA-attrs
- https://huggingface.co/datasets/phiyodr/coco2017
- https://huggingface.co/microsoft/resnet-50
- https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop
