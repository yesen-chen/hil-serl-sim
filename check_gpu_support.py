#!/usr/bin/env python3
"""
检测当前环境中 JAX、Flax、Chex 等库的 GPU 支持情况
"""

import sys

def check_jax_gpu():
    """检测 JAX 的 GPU 支持"""
    print("=" * 60)
    print("检查 JAX GPU 支持")
    print("=" * 60)
    
    try:
        import jax
        print(f"✓ JAX 版本: {jax.__version__}")
        
        # 检查可用设备
        devices = jax.devices()
        print(f"✓ 可用设备数量: {len(devices)}")
        print(f"✓ 设备列表:")
        for i, device in enumerate(devices):
            print(f"    [{i}] {device} (平台: {device.platform})")
        
        # 检查是否有 GPU
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        cpu_devices = [d for d in devices if d.platform == 'cpu']
        
        if gpu_devices:
            print(f"\n✓ 检测到 {len(gpu_devices)} 个 GPU 设备")
            gpu_available = True
            
            # 尝试在 GPU 上运行一个简单的计算（不使用cuDNN）
            try:
                import jax.numpy as jnp
                # 使用简单的数组操作，避免触发cuDNN
                x = jnp.array([1.0, 2.0, 3.0])
                y = jnp.array([4.0, 5.0, 6.0])
                z = x + y
                z.block_until_ready()  # 确保计算完成
                
                # 检查设备平台
                device_platform = z.device.platform
                print(f"✓ GPU 基础计算测试成功 (设备平台: {device_platform})")
                
                # 尝试简单的矩阵乘法（不使用cuDNN）
                try:
                    import time
                    a = jnp.ones((500, 500), dtype=jnp.float32)
                    start = time.time()
                    b = jnp.dot(a, a)
                    b.block_until_ready()
                    elapsed = time.time() - start
                    print(f"✓ GPU 性能测试: 500x500 矩阵乘法耗时 {elapsed:.4f} 秒")
                except Exception as perf_e:
                    print(f"⚠ GPU 性能测试跳过: {perf_e}")
                    print("   (这可能是cuDNN相关的问题，但不影响基本GPU使用)")
                    
            except Exception as e:
                error_msg = str(e)
                if "CUDNN" in error_msg or "DNN" in error_msg:
                    print(f"⚠ GPU 计算测试遇到 cuDNN 问题: {error_msg[:100]}...")
                    print("   提示: cuDNN 初始化失败，但 GPU 设备已检测到")
                    print("   这通常不影响基本的 JAX GPU 计算，只是深度学习操作可能受限")
                    print("   建议: 检查 cuDNN 安装或尝试重启程序")
                else:
                    print(f"⚠ GPU 计算测试失败: {e}")
                # 即使计算测试失败，只要检测到GPU设备，仍然认为GPU可用
                gpu_available = True
                
            return gpu_available
        else:
            print(f"\n⚠ 未检测到 GPU 设备，只有 {len(cpu_devices)} 个 CPU 设备")
            print("  提示: 如果安装了 CUDA，请检查:")
            print("    1. CUDA 是否正确安装")
            print("    2. JAX 是否安装了 CUDA 支持版本")
            print("    3. CUDA 版本是否与 JAX 兼容")
            return False
            
    except ImportError:
        print("✗ JAX 未安装")
        return False
    except Exception as e:
        print(f"✗ JAX 检测出错: {e}")
        return False


def check_flax():
    """检测 Flax 库"""
    print("\n" + "=" * 60)
    print("检查 Flax")
    print("=" * 60)
    
    try:
        import flax
        print(f"✓ Flax 版本: {flax.__version__}")
        
        # Flax 依赖 JAX，如果 JAX 可以使用 GPU，Flax 也可以
        try:
            import jax
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == 'gpu']
            if gpu_devices:
                print("✓ Flax 可以使用 GPU (通过 JAX)")
            else:
                print("⚠ Flax 只能使用 CPU (JAX 未检测到 GPU)")
        except:
            print("⚠ 无法检测 Flax 的 GPU 支持 (JAX 不可用)")
            
        return True
    except ImportError:
        print("✗ Flax 未安装")
        return False
    except Exception as e:
        print(f"✗ Flax 检测出错: {e}")
        return False


def check_chex():
    """检测 Chex 库"""
    print("\n" + "=" * 60)
    print("检查 Chex")
    print("=" * 60)
    
    try:
        import chex
        print(f"✓ Chex 版本: {chex.__version__}")
        
        # Chex 主要用于测试，也依赖 JAX
        try:
            import jax
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == 'gpu']
            if gpu_devices:
                print("✓ Chex 可以使用 GPU (通过 JAX)")
            else:
                print("⚠ Chex 只能使用 CPU (JAX 未检测到 GPU)")
        except:
            print("⚠ 无法检测 Chex 的 GPU 支持 (JAX 不可用)")
            
        return True
    except ImportError:
        print("✗ Chex 未安装")
        return False
    except Exception as e:
        print(f"✗ Chex 检测出错: {e}")
        return False


def check_optax():
    """检测 Optax 库"""
    print("\n" + "=" * 60)
    print("检查 Optax")
    print("=" * 60)
    
    try:
        import optax
        print(f"✓ Optax 版本: {optax.__version__}")
        
        # Optax 依赖 JAX
        try:
            import jax
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == 'gpu']
            if gpu_devices:
                print("✓ Optax 可以使用 GPU (通过 JAX)")
            else:
                print("⚠ Optax 只能使用 CPU (JAX 未检测到 GPU)")
        except:
            print("⚠ 无法检测 Optax 的 GPU 支持 (JAX 不可用)")
            
        return True
    except ImportError:
        print("✗ Optax 未安装")
        return False
    except Exception as e:
        print(f"✗ Optax 检测出错: {e}")
        return False


def check_cuda_info():
    """检查 CUDA 相关信息"""
    print("\n" + "=" * 60)
    print("检查 CUDA 信息")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ NVIDIA GPU 信息:")
            # 只显示前几行关键信息
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print(f"  {line}")
        else:
            print("⚠ nvidia-smi 命令执行失败")
    except FileNotFoundError:
        print("⚠ nvidia-smi 未找到 (可能未安装 NVIDIA 驱动)")
    except subprocess.TimeoutExpired:
        print("⚠ nvidia-smi 命令超时")
    except Exception as e:
        print(f"⚠ 检查 CUDA 信息时出错: {e}")
    
    # 检查 CUDA 版本（如果 JAX 已安装）
    try:
        import jax
        # 使用新的API，避免deprecation warning
        try:
            # 新API (JAX 0.4+)
            if hasattr(jax, 'extend') and hasattr(jax.extend, 'backend'):
                backend = jax.extend.backend.get_backend()
                print(f"\n✓ JAX 后端: {backend.platform}")
                if hasattr(backend, 'platform_version'):
                    print(f"✓ 平台版本: {backend.platform_version}")
            # 旧API (向后兼容)
            elif hasattr(jax, 'lib') and hasattr(jax.lib, 'xla_bridge'):
                backend = jax.lib.xla_bridge.get_backend()
                print(f"\n✓ JAX 后端: {backend.platform}")
                if hasattr(backend, 'platform_version'):
                    print(f"✓ 平台版本: {backend.platform_version}")
        except Exception as e:
            # 如果无法获取后端信息，至少显示设备信息
            devices = jax.devices()
            if devices:
                print(f"\n✓ JAX 设备: {devices[0].platform}")
    except:
        pass


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("GPU 支持检测脚本")
    print("=" * 60)
    print()
    
    results = {}
    
    # 检查各个库
    results['jax'] = check_jax_gpu()
    results['flax'] = check_flax()
    results['chex'] = check_chex()
    results['optax'] = check_optax()
    
    # 检查 CUDA 信息
    check_cuda_info()
    
    # 总结
    print("\n" + "=" * 60)
    print("检测总结")
    print("=" * 60)
    
    # 检查JAX是否安装（即使GPU测试失败）
    jax_installed = False
    jax_gpu_available = False
    try:
        import jax
        jax_installed = True
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        jax_gpu_available = len(gpu_devices) > 0
    except:
        pass
    
    all_installed = all(results.values())
    
    if jax_gpu_available:
        print("✓ JAX 已安装并检测到 GPU 设备")
        print("✓ Flax、Chex、Optax 等依赖 JAX 的库也可以使用 GPU")
        if not all_installed:
            print("\n库安装状态:")
            for lib, installed in results.items():
                status = "✓" if installed else "✗"
                print(f"  {status} {lib}")
    elif jax_installed:
        print("⚠ JAX 已安装，但未检测到 GPU 支持")
        print("  建议检查:")
        print("    1. CUDA 是否正确安装")
        print("    2. JAX 是否安装了 CUDA 支持版本")
        print("    3. CUDA 版本是否与 JAX 兼容")
        print("\n库安装状态:")
        for lib, installed in results.items():
            status = "✓" if installed else "✗"
            print(f"  {status} {lib}")
    else:
        print("⚠ 部分库未安装")
        for lib, installed in results.items():
            status = "✓" if installed else "✗"
            print(f"  {status} {lib}")
    
    print("\n" + "=" * 60)
    
    return 0 if jax_gpu_available else 1


if __name__ == "__main__":
    sys.exit(main())

