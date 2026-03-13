import os
import argparse
from src.pipeline import UCPOFPipeline

def main():
    """提取特征并保存CSV"""
    parser = argparse.ArgumentParser(description='Extract features from LLM outputs')
    parser.add_argument('--dataset-config', type=str, default='configs/dataset/ace.yaml', help='Path to dataset configuration file')
    parser.add_argument('--model-config', type=str, default='configs/model/llama3_8b.yaml', help='Path to model configuration file')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=5000, help='Number of samples to evaluate')
    args = parser.parse_args()
    
    # 初始化管道
    pipeline = UCPOFPipeline(
        dataset_config_path=args.dataset_config,
        model_config_path=args.model_config,
        output_dir=args.output_dir
    )
    
    # 运行离线流程，提取特征
    df = pipeline.run_offline(num_samples=args.num_samples)
    
    if df is not None:
        print("Feature extraction completed successfully!")
    else:
        print("Feature extraction failed!")

if __name__ == '__main__':
    main()
