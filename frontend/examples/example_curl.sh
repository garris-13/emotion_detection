#!/bin/bash

# 表情识别 API cURL 使用示例

echo "======================================================================"
echo "表情识别 API - cURL 使用示例"
echo "======================================================================"

API_URL="http://localhost:5000"

# 1. 检查 API 健康状态
echo -e "\n【示例 1】检查 API 健康状态"
echo "----------------------------------------------------------------------"
echo "命令: curl ${API_URL}/health"
curl -s ${API_URL}/health | python -m json.tool

# 2. 获取支持的表情列表
echo -e "\n【示例 2】获取支持的表情列表"
echo "----------------------------------------------------------------------"
echo "命令: curl ${API_URL}/emotions"
curl -s ${API_URL}/emotions | python -m json.tool

# 3. 获取 API 信息
echo -e "\n【示例 3】获取 API 信息"
echo "----------------------------------------------------------------------"
echo "命令: curl ${API_URL}/"
curl -s ${API_URL}/ | python -m json.tool

# 4. 单图预测 - 文件上传方式
echo -e "\n【示例 4】单图预测 - 文件上传方式"
echo "----------------------------------------------------------------------"
IMAGE_PATH="test_image.jpg"
if [ -f "$IMAGE_PATH" ]; then
    echo "命令: curl -X POST ${API_URL}/predict -F \"image=@${IMAGE_PATH}\""
    curl -s -X POST ${API_URL}/predict -F "image=@${IMAGE_PATH}" | python -m json.tool
else
    echo "跳过: 测试图像不存在 (${IMAGE_PATH})"
fi

# 5. 单图预测 - Base64方式
echo -e "\n【示例 5】单图预测 - Base64方式"
echo "----------------------------------------------------------------------"
if [ -f "$IMAGE_PATH" ]; then
    IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH" 2>/dev/null || base64 "$IMAGE_PATH")
    echo "命令: curl -X POST ${API_URL}/predict -H \"Content-Type: application/json\" -d '{\"image\": \"...\"}'"
    curl -s -X POST ${API_URL}/predict \
      -H "Content-Type: application/json" \
      -d "{\"image\": \"${IMAGE_BASE64}\"}" | python -m json.tool
else
    echo "跳过: 测试图像不存在 (${IMAGE_PATH})"
fi

# 6. 批量预测
echo -e "\n【示例 6】批量预测"
echo "----------------------------------------------------------------------"
IMAGE1="test1.jpg"
IMAGE2="test2.jpg"
IMAGE3="test3.jpg"

if [ -f "$IMAGE1" ] && [ -f "$IMAGE2" ]; then
    echo "命令: curl -X POST ${API_URL}/predict_batch -F \"images=@${IMAGE1}\" -F \"images=@${IMAGE2}\""
    curl -s -X POST ${API_URL}/predict_batch \
      -F "images=@${IMAGE1}" \
      -F "images=@${IMAGE2}" \
      -F "images=@${IMAGE3}" | python -m json.tool
else
    echo "跳过: 测试图像不存在"
fi

echo -e "\n======================================================================"
echo "示例运行完成"
echo "======================================================================"

# 使用说明
echo -e "\n使用说明:"
echo "1. 确保 API 服务器正在运行 (python backend/api/api_server.py)"
echo "2. 准备测试图像文件"
echo "3. 运行此脚本: bash frontend/examples/example_curl.sh"
echo ""
echo "Windows 用户提示:"
echo "- PowerShell 用户可以使用 Invoke-WebRequest 或 Invoke-RestMethod"
echo "- Git Bash 用户可以直接运行此脚本"
echo "======================================================================"
