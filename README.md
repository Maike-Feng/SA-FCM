# *A Two-branch Network with Spatial-structure Encoding and Subpixel Aggregation for Hyperspectral Image Classification*

## **Abstract**
Most hyperspectral image (HSI) classification methods perform well in extracting spectral sequence features and spatial semantic features. However, their abilities to represent spatial structural features are still insufficient. This limitation makes the enhancement of spatial features insufficient. Additionally, features at the subpixel level have not been fully utilized, leading to an unsatisfactory extraction effect for detailed information. To address these issues, this paper proposes a dual branch network with spatial-structure encoding and subpixel aggregation (SESA) to achieve HSI classification. It contains two modules: a spatial-structure encoding (SSE) module and a subpixel aggregation (SPA) module. The SSE module extracts the structural features of HSIs by considering the proximity of pixels at different positions in the image, while the SPA module captures edge texture information through subpixel features. The
output features of the two modules are fused through weighted fusion, comprehensively enhancing the classification performance of the model. Experiments show that the proposed method outperforms other methods on multiple
datasets, and is especially superior in the smoothness of large objects and detail preservation of scattered objects, consuming less training and inference time.

## **Tool**:
- Python
- Pytorch
- Anaconda

---

## **How to Run**
1. Modify the dataset path in the `Data_Loader.py` file.
2. Run the `main.py` file.

## **Public dataset link**
- [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)
- [WHU-Hi UAV-borne Hyperspectral Datasets](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm)

## **Citation**
If the code here is useful to you, please cite this article.
