#!/bin/bash

set -e  # 出现错误时立即退出

# 容器名称，与 docker-compose.yml 中保持一致
CONTAINER_NAME="cz"

echo "🔧 [1/3] 构建 Docker 镜像..."
# 使用 docker-compose 构建镜像
docker compose -f docker-compose.yml build

echo "🧹 [2/3] 清理无用镜像缓存..."
# 删除未被使用的中间镜像、dangling images
docker system prune -f

echo "🚀 [3/3] 启动容器..."
# 使用 docker-compose 启动容器，后台运行
docker compose -f docker-compose.yml up -d

echo "🎉 完成！进入容器："
echo "    docker exec -it ${CONTAINER_NAME} bash"
