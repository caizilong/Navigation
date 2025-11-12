#!/usr/bin/env python3
"""
Mamba环境验证脚本
测试所有必需的依赖是否正确安装并可用
"""

import sys
import os

def test_basic_imports():
    """测试基础库导入"""
    print("="*60)
    print("测试 1: 基础库导入")
    print("="*60)
    
    tests = []
    
    # PyTorch相关
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"✓ torch {torch.__version__} (CUDA: {cuda_available})")
        if cuda_available:
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - GPU数量: {torch.cuda.device_count()}")
            print(f"  - GPU 0: {torch.cuda.get_device_name(0)}")
        tests.append(True)
    except Exception as e:
        print(f"✗ torch 导入失败: {e}")
        tests.append(False)
    
    # Mamba核心
    try:
        import mamba_ssm
        print(f"✓ mamba-ssm")
        tests.append(True)
    except Exception as e:
        print(f"✗ mamba-ssm 导入失败: {e}")
        tests.append(False)
    
    # causal-conv1d
    try:
        import causal_conv1d
        print(f"✓ causal-conv1d")
        tests.append(True)
    except Exception as e:
        print(f"✗ causal-conv1d 导入失败: {e}")
        tests.append(False)
    
    # einops
    try:
        from einops import rearrange, repeat
        print(f"✓ einops")
        tests.append(True)
    except Exception as e:
        print(f"✗ einops 导入失败: {e}")
        tests.append(False)
    
    # pytorch-lightning
    try:
        import pytorch_lightning as pl
        print(f"✓ pytorch-lightning {pl.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"✗ pytorch-lightning 导入失败: {e}")
        tests.append(False)
    
    # huggingface_hub
    try:
        import huggingface_hub
        print(f"✓ huggingface_hub")
        tests.append(True)
    except Exception as e:
        print(f"✗ huggingface_hub 导入失败: {e}")
        tests.append(False)
    
    return all(tests)


def test_mamba_components():
    """测试Mamba组件导入"""
    print("\n" + "="*60)
    print("测试 2: Mamba组件导入")
    print("="*60)
    
    tests = []
    
    # Block
    try:
        from mamba_ssm.modules.block import Block
        print(f"✓ mamba_ssm.modules.block.Block")
        tests.append(True)
    except Exception as e:
        print(f"✗ Block 导入失败: {e}")
        tests.append(False)
    
    # Triton ops (可选,可能失败)
    try:
        from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
        print(f"✓ mamba_ssm.ops.triton.layernorm_gated")
        tests.append(True)
    except Exception as e:
        print(f"⚠ Triton ops 不可用 (可选): {e}")
        tests.append(True)  # 不影响主要功能
    
    # 分布式工具 (可选)
    try:
        from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        print(f"✓ mamba_ssm.distributed")
        tests.append(True)
    except Exception as e:
        print(f"⚠ 分布式工具不可用 (可选): {e}")
        tests.append(True)  # 不影响主要功能
    
    return all(tests)


def test_mamba_forward_pass():
    """测试Mamba前向传播"""
    print("\n" + "="*60)
    print("测试 3: Mamba前向传播 (CPU)")
    print("="*60)
    
    try:
        import torch
        from mamba_ssm.modules.block import Block
        
        # 配置
        d_model = 128
        batch_size = 2
        seq_len = 16
        
        print(f"配置: d_model={d_model}, batch={batch_size}, seq_len={seq_len}")
        
        # 创建输入
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"✓ 输入创建成功: {x.shape}")
        
        # 创建Mamba block
        # 注意: 这里使用默认的mixer_cls,实际使用时需要指定Mamba2
        block = Block(dim=d_model)
        print(f"✓ Block创建成功")
        
        # 前向传播
        output = block(x)
        print(f"✓ 前向传播成功: {output.shape}")
        
        # 验证输出shape
        assert output.shape == x.shape, f"输出shape不匹配: {output.shape} vs {x.shape}"
        print(f"✓ 输出shape验证通过")
        
        return True
        
    except Exception as e:
        print(f"✗ Mamba前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mamba_cuda():
    """测试Mamba CUDA加速"""
    print("\n" + "="*60)
    print("测试 4: Mamba CUDA加速 (可选)")
    print("="*60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("⚠ CUDA不可用,跳过GPU测试")
            return True
        
        from mamba_ssm.modules.block import Block
        
        # 配置
        d_model = 256
        batch_size = 4
        seq_len = 32
        
        device = torch.device("cuda:0")
        print(f"使用设备: {device}")
        
        # 创建输入
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        print(f"✓ GPU输入创建成功: {x.shape}")
        
        # 创建Mamba block
        block = Block(dim=d_model).to(device)
        print(f"✓ Block移至GPU成功")
        
        # 前向传播
        output = block(x)
        print(f"✓ GPU前向传播成功: {output.shape}")
        
        # 清理显存
        del x, output, block
        torch.cuda.empty_cache()
        print(f"✓ 显存清理完成")
        
        return True
        
    except Exception as e:
        print(f"⚠ GPU测试失败 (不影响CPU功能): {e}")
        return True  # GPU测试失败不算错误


def test_existing_dependencies():
    """测试现有visualnav依赖"""
    print("\n" + "="*60)
    print("测试 5: visualnav-transformer现有依赖")
    print("="*60)
    
    tests = []
    
    deps = [
        'torchvision',
        'numpy',
        'matplotlib',
        'opencv-python',
        'h5py',
        'wandb',
        'efficientnet_pytorch',
        'diffusers',
        'tqdm',
    ]
    
    for dep in deps:
        module_name = dep.replace('-', '_')
        try:
            __import__(module_name)
            print(f"✓ {dep}")
            tests.append(True)
        except Exception as e:
            print(f"✗ {dep} 不可用: {e}")
            tests.append(False)
    
    return all(tests)


def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("Mamba依赖环境验证")
    print("="*60)
    print(f"Python版本: {sys.version}")
    print(f"当前目录: {os.getcwd()}")
    print("")
    
    results = []
    
    # 运行测试
    results.append(("基础库导入", test_basic_imports()))
    results.append(("Mamba组件导入", test_mamba_components()))
    results.append(("Mamba前向传播", test_mamba_forward_pass()))
    results.append(("Mamba CUDA", test_mamba_cuda()))
    results.append(("现有依赖", test_existing_dependencies()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print("")
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✅ 所有测试通过! 环境配置正确,可以开始集成工作")
        return 0
    else:
        print("\n⚠️ 部分测试失败,请检查错误信息并重新安装相关依赖")
        return 1


if __name__ == "__main__":
    sys.exit(main())
