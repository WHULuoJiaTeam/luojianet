# graph engine 个人开发工具链使用说明

GE开发者工具链是graph engine中的一套面向个人开发者的自动化脚本工具链。

目前支持基于容器开发环境准备、构建依赖的自动下载安装和配置、代码格式化、编译、测试、代码覆盖率检查、文档生成等一系列开发者常用功能。

## 前置准备

下面是使用GE开发者工具链，需要手动进行的前置准备；

下列是经过验证后的性能最佳推荐配置：

1. 操作系统，以下任选其一：
    - 原生的Linux操作系统，如ubuntu；
    - Windows操作系统，推荐安装WSL的ubuntu系统，强烈建议登录WSL内直接下载代码，不要挂卷（构建性能差）！
    - MAC OS；

2. docker安装：
    - docker安装成功，并且相关镜像源已经设置正确，可正常下载外部镜像。

3. OS支持的命令行工具： 原生Linux or WSL shell；

可用但不推荐的配置：

- 在windows中直接安装docker，采用仿linux bash（Cygwin，minGW等）执行ge工具链；
  （使用这种方式也可以执行所有GE工具链的操作，但是因为windows和容器异构内核的文件访问限制会导致构建速度比较慢）

## 快速上手

GE工具链对应的脚本在scripts下，可以按照下面流程来执行：

1. 进入到scripts目录：

```sh
$ cd ./scripts
```

2.执行`ge env`自动下载容器环境，并登陆到环境中

```sh
$ ./ge.sh env 
```

3.下载和安装构建所依赖的外部库 
    ```sh
    $ ge update
    ```
（注：进入容器后，`ge`命令已经自动注册进系统，因此容器内不需要写脚本全称）

4.执行测试，默认执行单元测试用例，`ge test`会自动触发构建

```sh
$ ge test
```

## 详细用法

在scripts目录下，运行./ge.sh -h 即可查看到所有的子命令集合。

```sh
$ ./ge.sh -h

Usage: ge  COMMANDS

Run ge commands

Commands:
    env         Prepare docker env
    config      Config dependencies server
    update      Update dependencies
    format      Format code
    lint        Static verify
    build       Build code
    test        Run test of UT/ST
    cov         Run Coverage
    docs        Generate documents
    clean       Clean
```

脚本内置的每个子命令，代表一个独立功能；每个子命令还提供了二级参数用于灵活指定执行方式。

每个子命令可以通过`-h`查看支持的可配参数。

例如查询`env`子命令的参数，可以使用如下命令：

```sh
$  ./ge.sh env -h
```

每个子命令在不带参数时，会有一个默认的行为。

### `ge env`

该命令用于准备构建和测试使用的容器环境，具体包含参数如下：

```
$  ./ge.sh env -h

Usage: ge env [OPTIONS]

Prepare for docker env for build and test

Options:
    -b, --build  Build docker image
    -p, --pull   Pull  docker image
    -e, --enter  Enter container
    -r, --reset  Reset container
    -h, --help
```

参数详细解释：
- `-b  -- build`： 依据“scripts/env/Dockerfile”生成需要运行的容器镜像；
- `-p  -- pull` ： 从本地配置的容器中央仓拉取需要的的容器镜像；
- `-e  -- enter`： 在本地已有容器镜像的前提下，登录容器运行环境；
- `-r  -- reset`： 删除本地运行的容器镜像环境；

默认：从中央容器仓拉取对应的容器镜像，运行实例并登陆。

### `ge config`

配置外部依赖服务器，具体参数如下：

```sh
$ ge config -h

Usage: ge config [OPTIONS]

update server config for ge, you need input all config info (ip, user, password)

Options:
    -i, --ip           Config ip config
    -u, --user         Config user name
    -p, --password     Config password
    -h, --help

Example: ge config -i=<ip-adress> -u=<username> -p=<password> (Need add escape character \ before special charactor $、#、!)
```

参数详细解释：

- `-i,  --ip`          : 配置依赖库服务器IP地址；
- `-u,  --usr`         : 配置依赖库服务器用户名；
- `-p,  --password`    : 配置依赖库地址；

默认：打印帮助信息。


### `ge update`

安装graph engine构建所需的外部依赖库，具体参数如下：

```sh
$ ge update -h

Usage: ge update [OPTIONS]

update dependencies of build and test

Options:
    -p, --public       Download dependencies from community
    -d, --download     Download dependencies
    -i, --install      Install dependencies
    -c, --clear        Clear dependencies
    -h, --help
```

参数详细解释：
- `-p,  --public`   : 从社区下载安装依赖库；
- `-d,  --download` : 下载构建需要外部依赖库；
- `-i,  --install`  : 安装外部依赖包到对应位置；
- `-c,  --clear`    : 清除下载的外部依赖包；

默认：根据"scripts/update/deps_config.sh"的配置下载外部依赖库并安装到对应目录。
（注：请确保“scripts/update/server_config.sh”中的服务器地址、用户名、密码已经配置）

### `ge format`

使用clang-format进行代码格式化，具体参数如下：

```sh
$ ge format -h
Options:
    -a format of all files
    -c format of the files changed compared to last commit, default case
    -l format of the files changed in last commit
    -h Print usage
``` 

参数详细解释：

- `-a` : 格式化所有代码；
- `-c` : 只格式化本次修改的代码；
- `-l` : 格式化上次提交的代码；

默认：格式化本次修改代码。

### `ge build`

执行构建 (注：调用原有build.sh脚本，改造中...)；

### `ge test`

构建和运行测试用例，目前可以支持参数如下：

```sh
$ ge test -h

Usage: ge test [OPTIONS]

Options:
    -u, --unit          Run unit Test
    -c, --component     Run component Test
    -h, --help
```

参数详细解释：

- `-u, --unit`      : 执行单元测试
- `-c, --component` : 执行组件测试

默认：执行单元测试。

### `ge cov`

执行代码覆盖率检查, 支持全量覆盖和增量覆盖的功能，该命令需要已经跑完测试用例，目前可以支持参数如下：

```sh
$ ge cov -h

Usage: ge cov [OPTIONS]

Options:
    -a, --all          Full coverage
    -i, --increment    Increment coverage
    -d, --directory    Coverage of directory
    -h, --help
```

参数详细解释：

- `-a, --all`       : 执行全量覆盖率统计；
- `-i, --increment` : 执行增量覆盖率检查，默认是分析未提交修改的代码覆盖率（如果存在新增加的git未跟踪文件，需要先git add 添加进来才可以）；
- `-d, --directory` : 代码进行增量覆盖率检查的代码路径，支持传入路径参数；

默认：执行增量覆盖率检查；

下面的命令演示了如何检查ge目录下所有代码的增量覆盖率：

```sh
$ ge cov -d=ge 
```

### `ge docs`

Doxygen文档生成，包含代码逻辑和物理结构和关系，方便阅读和理解代码；目前可以支持参数如下：

```sh
$ ge docs -h

Usage: ge docs [OPTIONS]

Options:
    -b, --brief  Build brief docs
    -a, --all    Build all docs
    -h, --help
```

参数详细解释：

- `-b, --brief` : 生成简要文档，忽略部分关系图生成，速度快；
- `-a, --all`   : 生成全量文档，包含各种代码关系图，速度相对慢；

默认： 生成全量代码文档。

### `ge clean`

清除各种下载或生成的中间文件，目前支持的参数如下：

```sh
$ ge clean -h

Usage: ge clean [OPTIONS]

Options:
    -b, --build         Clean build dir
    -d, --docs          Clean generate docs
    -i, --install       Clean dependenices
    -a, --all           Clean all
    -h, --help
```

参数详细解释：

- `-b, --build`   : 清除生成的编译构建临时文件；
- `-d, --docs`    : 清除生成的文档临时文件；
- `-i, --install` : 清除安装的依赖文件；
- `-a, --all`     : 清除所有下载和生成的临时文件；

默认：清除编译构建产生临时文件。

## Follow us

工具链的功能还在不断完善中，有问题请提issue，谢谢！
