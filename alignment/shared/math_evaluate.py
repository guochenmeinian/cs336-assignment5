#!/usr/bin/env python3
"""
通用模型评估脚本 - 可以评估任何训练后的模型
"""

import logging
from pathlib import Path

from .math_evaluation_utils import evaluate_vllm, load_model
from .config import EVALUATIONS_DIR, MATH_TEST_PATH, DEFAULT_EVAL_SAMPLING_PARAMS
from .math_data_utils import load_math_data, format_r1_zero_prompt
from .drgrpo_grader import r1_zero_reward_fn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, output_name: str = None):
    """评估指定路径的模型"""
    
    # 如果没有指定输出名称，自动生成包含模型信息的名称
    if output_name is None:
        model_path_obj = Path(model_path)
        # 提取模型目录名，去掉可能的policy后缀
        model_dir_name = model_path_obj.name
        if model_dir_name == "policy":
            model_dir_name = model_path_obj.parent.name
        
        # 生成描述性的输出名称
        output_name = f"{model_dir_name}_evaluation"
    
    # 输出路径
    output_path = EVALUATIONS_DIR / f"{output_name}.jsonl"
    
    # 确保输出目录存在
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 使用默认采样参数
    from vllm import SamplingParams
    eval_sampling_params = SamplingParams(**DEFAULT_EVAL_SAMPLING_PARAMS)
    
    logger.info(f"加载模型: {model_path}")
    
    # 加载模型
    try:
        # 如果是本地模型路径，直接加载
        if Path(model_path).exists():
            from vllm import LLM
            model = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.5
            )
            logger.info("本地模型加载成功")
        else:
            # 尝试作为模型名称加载
            model = load_model(model_path)
            logger.info("预训练模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.error(f"请检查模型路径: {model_path}")
        logger.error("SFT模型通常保存在: models/sft_xxx/policy/ 目录下")
        logger.error("请使用完整路径，例如: models/sft_qwen_math_15b_size128_lr2e-05_bs2_steps100/policy")
        return False
    
    # 加载测试数据
    test_data_path = str(MATH_TEST_PATH)
    logger.info(f"加载测试数据: {test_data_path}")
    test_data = load_math_data(test_data_path)
    
    # 格式化prompts
    logger.info("使用r1_zero模板格式化prompts...")
    prompts = []
    ground_truths = []
    
    for example in test_data:
        question = example["question"]
        answer = example["answer"]
        
        # 从<answer>...</answer>格式提取纯答案
        if "<answer>" in answer and "</answer>" in answer:
            pure_answer = answer.split("<answer>")[-1].replace("</answer>", "").strip()
        else:
            logger.warning(f"意外的答案格式: {answer[:100]}...")
            pure_answer = answer
        
        # 使用r1_zero模板格式化prompt
        prompt = format_r1_zero_prompt(question)
        prompts.append(prompt)
        ground_truths.append(pure_answer)
    
    logger.info(f"格式化了 {len(prompts)} 个prompts")
    
    # 评估模型
    logger.info("开始评估...")
    evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        eval_sampling_params=eval_sampling_params,
        ground_truths=ground_truths,
        output_path=str(output_path),
    )
    
    logger.info("评估完成!")
    logger.info(f"结果保存到: {output_path}")
    
    return True


def main():
    """主函数 - 用于命令行调用"""
    import argparse
    
    parser = argparse.ArgumentParser(description="评估训练后的模型")
    parser.add_argument("model_path", help="模型路径或名称")
    parser.add_argument("--output", "-o", help="输出文件名（不包含.jsonl后缀）")
    
    args = parser.parse_args()
    
    # 运行评估
    success = evaluate_model(args.model_path, args.output)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
