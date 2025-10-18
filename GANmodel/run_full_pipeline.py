#!/usr/bin/env python
"""
GANmodel完整运行脚本
自动化运行整个GAN增强的信用风险预测流程
"""

import subprocess
import os
import sys
from datetime import datetime


def print_step(step_num, title):
    """打印步骤标题"""
    print("\n" + "="*70)
    print(f"  步骤 {step_num}: {title}")
    print("="*70 + "\n")


def run_command(cmd, description):
    """运行命令并显示输出"""
    print(f"▶ {description}")
    print(f"命令: {cmd}\n")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} 完成\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} 失败")
        print(f"错误: {e}\n")
        return False
    except Exception as e:
        print(f"✗ 执行出错: {e}\n")
        return False


def main():
    """主函数"""
    print("="*70)
    print("    GANmodel 完整流程自动化运行")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 配置参数
    GAN_EPOCHS = 50  # GAN训练轮数（可根据需要调整）
    GAN_BATCH_SIZE = 100
    NUM_SYNTHETIC = 200  # 生成合成样本数

    # 步骤1: 训练GAN模型
    print_step(1, "训练GAN模型生成合成数据")
    success = run_command(
        f"python train_gan.py "
        f"--epochs {GAN_EPOCHS} "
        f"--batch-size {GAN_BATCH_SIZE} "
        f"--use-sdv "
        f"--target-class minority",
        "训练GAN模型（只针对违约样本）"
    )

    if not success:
        print("⚠️  GAN训练失败，但可以继续后续步骤（如果已有模型）")

    # 步骤2: 检查GAN模型
    print_step(2, "检查GAN模型")
    if os.path.exists("models/sdv_ctgan_minority_latest.pkl"):
        print("✓ GAN模型已存在\n")
    else:
        print("✗ GAN模型不存在，后续步骤可能失败\n")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            print("流程中止")
            return 1

    # 步骤3: 使用GAN增强数据训练预测模型
    print_step(3, "使用GAN增强数据训练G-XGBoost模型")
    success = run_command(
        f"python train_with_gan.py "
        f"--gan-model-path models/sdv_ctgan_minority_latest.pkl "
        f"--num-synthetic {NUM_SYNTHETIC} "
        f"--augment-strategy minority "
        f"--model-type stacking "
        f"--output-name g_xgboost "
        f"--use-sdv",
        "训练G-XGBoost模型"
    )

    if not success:
        print("✗ G-XGBoost训练失败")
        return 1

    # 步骤4: 生成可视化
    print_step(4, "生成预测结果可视化")
    success = run_command(
        "python plot_predictions.py",
        "生成预测分布图和对比分析"
    )

    if not success:
        print("⚠️  可视化生成失败（可能是没有预测结果文件）")

    # 完成
    print("\n" + "="*70)
    print("  ✓ 完整流程运行完成!")
    print("="*70)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("生成的文件:")
    print("  - GAN模型: models/")
    print("  - 预测结果: outputs/g_xgboost_submission.csv")
    print("  - 可视化图表: outputs/plots/")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断运行")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
