import os
import argparse
import yaml
from src.pipeline import UCPOFPipeline
from analysis.plot_pareto_efficiency import plot_pareto_efficiency
from analysis.plot_risk_coverage import plot_risk_coverage
from analysis.plot_kde_distribution import plot_kde_distribution
from analysis.threshold_analysis import threshold_analysis
from analysis.metric_validation import metric_validation

def main():
    """运行完整的UCPOF实验"""
    parser = argparse.ArgumentParser(description='Run UCPOF experiment')
    parser.add_argument('--dataset-config', type=str, default='configs/dataset/ace.yaml', help='Path to dataset configuration file')
    parser.add_argument('--model-config', type=str, default='configs/model/llama3_8b.yaml', help='Path to model configuration file')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=5000, help='Number of samples to evaluate')
    parser.add_argument('--run-analysis', action='store_true', help='Run analysis after extraction')
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
        
        # 如果需要运行分析
        if args.run_analysis:
            # 获取模型名称
            with open(args.model_config, 'r') as f:
                model_config = yaml.safe_load(f)
            model_name = os.path.basename(os.path.normpath(model_config['model_path']))
            csv_path = os.path.join(args.output_dir, f"{model_name}_offline_metrics.csv")
            
            # 运行分析
            print("Running analysis...")
            plot_pareto_efficiency(csv_path, args.output_dir)
            plot_risk_coverage(csv_path, args.output_dir)
            plot_kde_distribution(csv_path, args.output_dir)
            threshold_analysis(csv_path, args.output_dir)
            metric_validation(csv_path, args.output_dir)
            print("Analysis completed successfully!")
    else:
        print("Feature extraction failed!")

if __name__ == '__main__':
    main()
