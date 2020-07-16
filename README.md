# Structure-Aware Human-Action Generation

We propose a variant of GCNs to leverage the powerful self-attention mechanism to adaptively sparsify a complete action graph in the temporal space. Our method could dynamically attend to important past frames and construct a sparse graph to apply in the GCN framework, well-capturing the structure information in action sequences. The paper can be found on arXiv [here!](https://arxiv.org/abs/2007.01971). ECCV 2020 accepted. 

If you use this code or these models, please cite the following paper:
```
@article{yu2020structure,
  title={Structure-Aware Human-Action Generation},
  author={Yu, Ping and Zhao, Yang and Li, Chunyuan and Chen, Changyou},
  journal={arXiv preprint arXiv:2007.01971},
  year={2020}
}
```
### Model Architecture
<p align="center">
  <img src=https://github.com/PingYu-iris/SA-GCN/blob/master/imgs/framework.jpg width="700" title="hover text">
</p>



### Prepare Dataset


