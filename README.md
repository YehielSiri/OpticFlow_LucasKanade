# Image-Pyramids-and-Optic-Flow
Implementation of Lucas Kanade algorithm, Image Alignment and Warping, Gaussian and Laplacian Pyramids.

All functions accept both gray-scale and color images as an input.

-----

1. [ Optical Flow Via Lucas Kanade ](#optical-flow-via-lucas-kanade)
2. [ Find Translation/Rigid Matrix ](#find-translationrigid-matrix)
3. [ Image Warping ](#image-warping)
4. [ Gaussian Pyramid ](#gaussian-pyramid)
5. [ Laplacian Pyramid ](#laplacian-pyramid)
6. [ Image Blending ](#image-blending)

-----

<h2>Optical Flow By Lucas Kanade</h2>
Accepting both gray-scale and color images as an input, but working on a gray-scale copy.
Calculate the Optical Flow by the Lucas Kanade algorithm which is optimized by the Iterative Algorithm.

<div align="center">

| Lucas Kanade Output |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>
 
 <div align="center">

| Hierarchical Lucas Kanade Output |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>
 
 <div align="center">

| Comparison (LK VS Hierarchical LK) |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>

-----

<h2>Find Translation/Rigid Matrix</h2>

 <div align="center">

| Find Translation Using LK |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>
 
 <div align="center">
 
| Find Rigid Using LK |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>
 
  <div align="center">

| Find Translation Using Correlation |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>
 
 <div align="center">
 
| Find Rigid Using Correlation |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>

-----

<h2>Image Warping</h2>

 <div align="center">
 
| Image Warping |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>

-----

<h2>Gaussian Pyramid</h2>
 <div align="center">
 
| Gaussian Pyramid |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>

-----

<h2>Laplacian Pyramid</h2>
 <div align="center">
 
| Laplacian Pyramid |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>
 
-----

<h2>Image Blending</h2>
 <div align="center">
 
| Image Blending |
| ------------- |
| <p align="center"><img src=""/></p>  |
  
 </div>
 
-----
