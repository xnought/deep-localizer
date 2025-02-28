<img src="https://github.com/user-attachments/assets/b00dc021-8e31-44be-9a69-ba33ed8054c6" width="800px">

**DeepLocalizer**: Find subnetworks in deep neural networks most active during a certain task/concept.

This library is meant to provide easy functions to accomplish :arrow_up: and provide evidence if we've actually found those subnetworks or not. 

Extends [The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units](https://arxiv.org/abs/2411.02280) to other models and data.

**Roadmap**

- [x] Replicate some parts of original paper (https://github.com/xnought/paper-implement/tree/main/language_network)
- [ ] Setup library to be used easily (uv) + connect to examples
- [ ] Set up face localizer example w/ goal of applying to a resnet model
- [ ] Basic API around pandas for data examples, with positive and negative/control examples
- [ ] Easy way to extract activations from the model
- [ ] Localization analysis API around activations of the data points given each model
- [ ] Visualize outputs
