import argparse
from modelscope.hub.snapshot_download import snapshot_download

def download_model(model_name, size):
    """下载指定的模型"""
    models = {
        "llama": {
            "8b": "LLM-Research/Meta-Llama-3.1-8B",
            "70b": "LLM-Research/Llama-3.3-70B-Instruct"
        },
        "qwen": {
            "1.5b": "Qwen/Qwen2.5-Math-1.5B"
        }
    }
    
    if model_name not in models:
        print(f"错误: 不支持的模型 '{model_name}'")
        print(f"支持的模型: {list(models.keys())}")
        return
    
    if size not in models[model_name]:
        print(f"错误: 模型 '{model_name}' 不支持大小 '{size}'")
        print(f"支持的大小: {list(models[model_name].keys())}")
        return
    
    model_path = models[model_name][size]
    print(f"开始下载 {model_name.upper()} {size} 模型...")
    print(f"模型路径: {model_path}")
    
    try:
        model_dir = snapshot_download(
            model_path,
            cache_dir='./autodl-tmp/'
        )
        print(f"✅ 模型下载完成！")
        print(f"📁 下载位置: {model_dir}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='下载指定的模型')
    parser.add_argument('model', choices=['llama', 'qwen'], help='选择模型类型')
    parser.add_argument('size', help='选择模型大小')
    
    args = parser.parse_args()
    
    # 显示可用的选项
    print("=" * 50)
    print("可用的模型和大小:")
    print("llama: 8b, 70b")
    print("qwen: 1.5b")
    print("=" * 50)
    
    download_model(args.model, args.size)

if __name__ == "__main__":
    main()
