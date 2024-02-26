&nbsp;&nbsp; This is project of 2022 spring CS492A: Machine Learning for 3D Data. Me and my teammate implemented Neural Parts from scratch.

&nbsp;&nbsp; Neural parts is one of the implicit representation methods, especially the primitive-based method. It introduces a new way of reconstructing the surface. The main idea is to use the learned shape as a primitive. The original paper[1] (Neural Parts: Learning Expressive 3D Shape Abstractions with Invertible Neural Networks) can be found [here][neuralpartspaper]. Though there were many hidden details that was not introduced in the paper, we have successfully reconstructed the code and got a similar result to the paper for the DFAUST dataset.

## Introduction

&nbsp;&nbsp;Lots of researchers have been worked on the task of reconstruction of surface from the given single image.
There are several kinds of surface representation, such as point cloud, mesh, voxel grid, and implicit function. This project focus on the primitive-based method, which is a kind of implicit function method. Primitive-based method represents the surface as a union of primitives, where primitive is a part of surface having simple geometry.

&nbsp;&nbsp;We have constructed the code of NeuralParts in our way and verified it using MPI Dynamic Faust dataset. As a first step, we preprocesed the data and build a data loader, which feeds preprocessed data as an input of a code. Our main code can be divided into three parts, model network, loss/metric function, and the train/test pipeline. We trained the model with a different number of primitives and then compared IOU and Chamfer-L1 distance with the paper result.

## Preprocessing

&nbsp;&nbsp;Before preprocessing the DFAUST dataset, we used provided code on DFAUST official website to get the mesh file. Three kinds of preprocessing are needed on the mesh.

&nbsp;&nbsp;First, we should render the mesh into 224 X 244 images. Those images will be used as input for our network. We rendered meshes in the same way as [Hierarchical Primitives[2]][hpgithub]

&nbsp;&nbsp;Second, we should sample points and normals on the surface of the mesh. To get surface samples, we sample 100,000 points from the mesh, and during training, we randomly sample 2,000 of them to calculate the loss. To sample points and normals from mesh, we used a library named trimesh.

&nbsp;&nbsp;Finally, we should sample points and labels uniformly distributed in the cube. The label is 1 if the point is inside the mesh and 0 if the point is outside the mesh. To get volume sample, we sample 100,000 points from space, and during training, we randomly sample 5,000 points, half of them inside the mesh and half of them outside the mesh, to calculate the loss. Objects are small compared to the cube, so we sample points inside the mesh with a higher probability so that sampled points have a similar number of in and out points. For unbiased loss, we should have to give weight to each point that is inversely proportional to its sampling probability. We referred to the code of [Occupancy Networks][ongithub][4] for the code segment to determine if the point is inside the mesh or not.

## Network

&nbsp;&nbsp;We have followed the network structure identical to the original paper. Overall network structure consists of two parts; Feature Extractor and Invertible Neural Network(INN). Feature Extractor captures the feature of the input image, and INN trains the homeomorphism between the sphere and each part.

&nbsp;&nbsp;Feature extractor is implemented in feature_extractor.py. First, image of size 224 $\times$ 224 is given as input of pretrained ResNet-18 layer given by pytorch. By concatenating this with the primitives, which is initialized as a random numbers and have size 256, we can construct a feature(denoted as $Cm$ in the implementation) which decides the behaviour of INN.

&nbsp;&nbsp;Invertible Neural Network(INN) in NeuralParts resembles the network structure introduced in [INN paper][innpaper][5]. INN is a stack of 4 conditional coupling layers, normalizer, and affine transformation layer. Each conditional coupling layer modifies one coordinate, and such coordinate is predefined. By passsing the other two coordinates to a network layer, we can decide how the modification will be done. For example, for the conditional coupling layer modifying $z$ coordinate, we have

$$ (x_o, y_o, z_o) = (x_i, y_i, t_\theta(x_i, y_i) + z_i \cdot exp(s_\theta(x_i, y_i))) $$

Note that its inverse is

$$ (x_i, y_i, z_i) = (x_o, y_o, (z_o - t_\theta(x_o, y_o)) \cdot exp(-s_\theta(x_o, y_o)) $$

so calculating inverse can be done by simply modifying the operation.

&nbsp;&nbsp;Note that in neuralParts, the homeomorphism between sphere and $m$th primitive is denoted as $\phi_\theta(\mathbf{x}, C_m)$, and implicit function representation of a surface is given as
$$g^m(\mathbf{x}) = \Vert \phi_\theta^{-1}(\mathbf{x};C_m)\Vert - r$$
where $r$ is a radius of sphere. Note that if we let $G$ by
$$G(\mathbf{x}) = \min_{1 \le m \le n_p} g^m(\mathbf{x}),$$
then $G(\mathbf{x}) < 0$ means that the point $\mathbf{x}$ is in the inside of a shape; $G(\mathbf{x}) = 0$ means that $\mathbf{x}$ is on a boundary surface, and $G(\mathbf{x}) > 0$ says $\mathbf{x}$ is on the outside.

## Loss and metric

&nbsp;&nbsp;The loss of our network is defined as a weighted sum of five loss functions, as follows.
$$ L  = w_{rec}L_{rec} + w_{occ}L_{occ} + w_{norm}L_{norm} \\
  + w_{overlap}L_{overlap} + w_{cover}L_{cover} $$
The model takes points randomly sampled from a sphere during training, giving surface points of predicted primitives as an output.
$L_{rec}$ is called reconstruction loss, and it is bidirectional Chamfer loss between surface samples and the output point.

&nbsp;&nbsp;$L_{occ}$ is called occupancy loss, and it checks whether labeled volume samples have the same inside/outside property for the predicted shape. We calculate this by using the value of $g^m$. Occupancy loss is cross-entropy classification loss by comparing the ground truth label and predicted label.

&nbsp;&nbsp;$L_{norm}$ is called normal consistency loss, and it measures how the unit normal vector of target shape and prediction by INN are different. Note that the surface is represented as an implicit function $G(\mathbf{x}) = r$, so $L_{norm}$ is calculated as an average of the value of $1 - \langle \nabla G(\mathbf{x}) / \Vert \nabla G(\mathbf{x}) \Vert, \mathbf{n}\rangle$ where $\mathbf{x}$ varies on the sample points and $\mathbf{n}$ is a ground truth unit normal.

&nbsp;&nbsp;$L_{overlap}$ is called overlapping loss, and it penalizes the overlapping of more than one primitives. For each sampled point $\mathbf{x}$, $L_{norm}$ can be calculated as an average of
$$max\left(0, \sum_{m = 1}^{n_p} \sigma(-g^m(\mathbf{x})/\tau) - \lambda\right).$$
Here the $\sigma$ is a sigmoid function, $\tau$ determines the sharpness, and $\lambda$ determines how much to penalize. In the original paper, $\lambda$ is setted to 1.95 as default.

&nbsp;&nbsp;$L_{cover}$ is called coverage loss, and it penalize too small primitives; so it maintains volume of each primitives. In the calculation of $L_{cover}$, the code of author is different from the formula in the Neural Parts.

&nbsp;&nbsp;IoU(Intersection over union) and Chamfer L1 distance were used as metrics to evaluate the model. IoU is the volume of the intersection divided by the volume of the union. To get IoU, we use preprocessed volume samples. (To remind, they are points sampled randomly from a cube.) We measure the volume of intersection and union by counting the number of volume samples inside, with considering weights. Getting Chamfer L1 distance is similar to the way getting reconstruction loss; we sum over the L1 distance of points.

## Hidden details in Implementation

&nbsp;&nbsp;This section describes some details were not described in the paper NeuralParts. We found them out by lots of trials.

&nbsp;&nbsp;Mesh rendering should be done with appropriate angles and contrast. Otherwise, the result is terrible. Man in DFAUST data move to the side, jump, and dance, so we should be able to know the x, y, and z position of the man by only one picture. So we should render mesh from a position somewhere in the top-right. It was tough finding out that rendering mesh in the same way the author did in [previous work][hpgithub] works well.

&nbsp;&nbsp;The paper suggests that points for volume sampling should be sampled uniformly from the unit cube. However, the mesh of DFAUST data is longer than 1m. So we decided to use 1 X 2 X 1 cuboid instead, considering the shape of human.

&nbsp;&nbsp;In INN, the output point of the stack of 4 coupling layers is not directly a points in a primitives. Although it is not described in paper, additional normalizing process and affine transformtion is needed. Normalizing process scales the point, and the scaling factor is given by elu function. Affine transformation layer translates and rotates the output point, and translation amount and rotation matrix are determined by passing 2-layer MLP with ReLU nonlinearlity.

&nbsp;&nbsp;Coverage loss design in github code was different from that of paper. Instead of the formula used in the original paper, author uses the formula
$$\sum_{m = 1}^{n_p}\sum_{\mathbf{x}} -\log(10^{-6} + \sigma(-g^m(\mathbf{x})/\tau)) )$$
as her code, where summation over $\mathbf{x}$ denotes the summation over the sampled points inside the primitives having least 10 values of $g^m$. Our code didn't work when we followed paper, so we followed the author's code instead.

## Experimental result

&nbsp;&nbsp;We trained the model with different number of primitives.

following table is result that we got:

|            | 2     | 5     | 8     | 10    |
| ---------- | ----- | ----- | ----- | ----- |
| IoU        | 0.640 | 0.654 | 0.645 | 0.643 |
| Chamfer-L1 | 0.109 | 0.092 | 0.089 | 0.089 |

following is the result of original paper:

|            | 2     | 5     | 8     | 10    |
| ---------- | ----- | ----- | ----- | ----- |
| IoU        | 0.675 | 0.673 | 0.676 | 0.678 |
| Chamfer-L1 | 0.101 | 0.097 | 0.090 | 0.095 |

&nbsp;&nbsp;Here is qualitative result with different number of primitives:
<img width="810" alt="result" src="https://github.com/buaaaaang/NeuralParts/assets/69184903/850baadd-202e-4b0f-8f13-460a2827decf">

we can observe that shape is represented well by primitives, but the alignment of primitives is different from that of original paper. The difference in alignment may be the reason for our result's different IoU and Chamfer-L1 distance.

&nbsp;&nbsp;In our observation, the alignment of primitives is fixed in the early stage of training by network initialization, and they are hard to be changed after the early stage. We guess that since there is high freedom in the form of primitives, the network tries to make each primitive more complex rather than fixing their alignment. This observation suggests that primitives that can be too complex are not always good.

## Conclusion

&nbsp;&nbsp;In summary, we have reconstructed the whole code of Neural Parts: data preprocess, model network, loss/metric functions, and train/test pipeline. We experimented using DFAUST dataset with different number of primitives and observe similar results with the paper. We also conducted an ablation study by removing one of the five losses and could get the same qualitative result with paper.

&nbsp;&nbsp;As mentioned before, from the output mesh of the trained model, we could observe that the alignment of primitive differs every time we train a new model. This may have happened by too much freedom in the shape of primitives. We suggest that finding a primitive-based method in the middle with less complex primitive than neural parts but still more exquisite primitive than convexes would give us a better primitive-based representation.

## How to use our code
To see mesh produced by trained network follow this instruction:

1. Select one of trained model from foler 'models'.
2. Open 'config.py' and set variable 'n_primitive' to be number of primitive of your selected model. (number of primitive of trained_model_n_prim is n)
3. On terminal, run 'python3 utils/result_visualization.py {path to selected model}'. for example, set n_primitive=5 and run 'python3 utils/result_visualization.py ./models/trained_model_5_prim.pth'.
4. Now open 'result' foler and you would see .obj file of overall mesh and each primitives, and .png file of rendered mesh.


Preprocessed data is about 300GB, so we cannot provide preprocessed data. If you want to test IOU and Chamfer-L1 loss of the model, or if you want to train the model yourself, follow this instruction:
1. Make sure you have more than 300GB of disk space.
2. Go to https://dfaust.is.tue.mpg.de/ and download 'MALE REGISTRATIONS' and 'FEMALE REGISTRATIONS'.
3. Move 'registrations_m.hdf5' and 'registrations_f.hdf5' into folder you want, and set variable 'dfaust_dataset_directory' of 'config.py' to be path of that folder.
4. Run 'python3 preprocess/preprocess_dfaust.py' on terimanl. Wait for perprocssing to finish. Preprocessing takes a long time.
5. For evaluation, select model, adjust 'n_primitives' of 'config.py', and run 'python3 test.py {path to selected model}'
6. for training, adjust parameters in 'config.py' and run 'python3 train.py'. The model will be saved as 'model.pth' at each step, and model with best validation score will be saved as 'best_model.pth'. 

## Acknowledgments

&nbsp;&nbsp;I worked on data preprocessing, building data loader, some loss/metric functions, and building an overall train/test pipeline, and debugging INN. Yuil studied and implemented feature extractor, Invertible Neural Network, and several loss functions and worked on ablation study.

&nbsp;&nbsp;As described before, to get DFAUST mesh, we used code in [here][dfaust]. To render mesh, we used render_dfaust.py in [here][hpgithub]. For labeling volume samples, we used inside_mesh.py in [here][ongithub]

## References

[1] Despoina Paschalidou, Angelos Katharopoulos, Andreas Geiger, and Sanja Fidler. Neural parts: Learning expressive 3d shape abstractions with invertible neural networks, 2021. \
[2] Despoina Paschalidou, Ali Osman Ulusoy, and Andreas Geiger. Superquadrics revisited: Learning 3d shape parsing beyond cuboids, 2019. \
[3] Federica Bogo, Javier Romero, Gerard Pons-Moll, and Michael J. Black. Dynamic FAUST: Registering human bodies in motion. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), July 2017. \
[4] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2019. \
[5] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp, 2017.

[neuralpartspaper]: https://arxiv.org/abs/2103.10429
[neuralpartsgithub]: https://github.com/buaaaaang/NeuralParts
[hpgithub]: https://github.com/paschalidoud/hierarchical_primitives
[ongithub]: https://github.com/autonomousvision/occupancy_networks

[INNPaper]: [https://arxiv.org/abs/1605.08803]
[dfaust]: [https://dfaust.is.tue.mpg.de/]
