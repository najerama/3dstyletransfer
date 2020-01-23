# Novel View Synthesis with Style Transfer via 3D feature embeddings

Recent years have seen progress in applying machine learning techniques to develop a 3D representation of a particular scene from 2D images. With the introduction of efficient 3D representations such as DeepVoxels, Scene Representation Networks (SRNs) and Neural Meshes, these neural networks are able to generate novel views of an object learned from a set of 2D images.

However, little work has been done in transforming these underlying 3D structures in meaningful ways. Can we, for instance, apply a new floral pattern to a 3D model of shape? We aim to explore the style transfer of a 2D image upon latent 3D models for novel view synthesis.

4 main contributions of our work: 
  1. We define a hardness scale for 3D Style Transfer for a dataset of 50 images 
  2. We extend Neural Meshes to ShapeNet's objects for future standardized research 
  3. We quantify loss with 3D style transfer network to backpropogate style features onto the underlying 3d scene representation. 
  4. We extend our approach to DeepVoxels and demonstrated the difficulty in both applying the network to new datasets as well as learning style transfer.


![](./misc/armchair_perry.gif)![](./misc/armchair_vg.gif)
