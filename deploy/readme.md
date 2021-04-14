# 基于Docker部署Tensorflow-serving服务

## 一、拉取镜像

###（一）、CPU

```commandline
docker pull tensorflow/serving
```

###（一）、GPU

在此之前需要先确保你的Docker支持gpu。

```commandline
docker pull tensorflow/serving:latest-gpu
```

## 二、启动容器

具体部署步骤可参考 [Tf-serving模型部署](https://www.yuque.com/docs/share/70c6ea9c-d35f-4175-ab35-92fb0995ad04) .