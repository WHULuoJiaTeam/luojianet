## LuoJiaNET Dockerfile Repository

This folder hosts all the `Dockerfile` to build LuoJiaNET container images with various hardware platforms.

### LuoJiaNET docker build command

| Hardware Platform | Version | Build Command |
| :---------------- | :------ | :------------ |
| CPU | `x.y.z` | cd luojianet_ms-cpu/x.y.z && docker build . -t luojianet_ms/luojianet_ms-cpu:x.y.z |
|  | `devel` | cd luojianet_ms-cpu/devel && docker build . -t luojianet_ms/luojianet_ms-cpu:devel |
|  | `runtime` | cd luojianet_ms-cpu/runtime && docker build . -t luojianet_ms/luojianet_ms-cpu:runtime |
| GPU | `x.y.z` | cd luojianet_ms-gpu/x.y.z  && docker build . -t luojianet_ms/luojianet_ms-gpu:x.y.z  |
|  | `devel` | cd luojianet_ms-gpu/devel && docker build . -t luojianet_ms/luojianet_ms-gpu:devel |
|  | `runtime` | cd luojianet_ms-gpu/runtime && docker build . -t luojianet_ms/luojianet_ms-gpu:runtime |

> **NOTICE:** The `x.y.z` version shown above should be replaced with the real version number.
