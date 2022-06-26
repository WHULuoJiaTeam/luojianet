## LuoJiaNET Conda build file Repository

This folder hosts all files relating to building conda packages, to download existing conda packages, simply typing
conda install -c luojianet_ms luojianet_ms-{platform}
in conda environments, whereby {platform} refers to hardware platform supported by LuoJiaNET, including CPU, GPU and Ascend

### LuoJiaNET conda install command

| Hardware Platform | Version | Download Command |
| :---------------- | :------ | :------------ |
| Ascend | `x.y.z` | conda install -c luojianet_ms luojianet_ms-ascend=x.y.z |
| CPU | `x.y.z` | conda install -c luojianet_ms luojianet_ms-cpu=x.y.z |
| GPU | `x.y.z` | conda install -c luojianet_ms luojianet_ms-gpu=x.y.z  |

> **NOTICE:** The `x.y.z` version shown above should be replaced with the real version number.
