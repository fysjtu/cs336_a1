import torch
import json
from pathlib import Path
from cs336_basics.transformer import TransformerBlock
from tests.adapters import run_transformer_block

def load_test_data():
    """加载测试数据和权重"""
    fixtures_path = Path("tests/fixtures")
    
    # 加载模型权重和配置
    state_dict = torch.load(fixtures_path / "ts_tests" / "model.pt", map_location="cpu")
    config = json.load(open(fixtures_path / "ts_tests" / "model_config.json"))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    return state_dict, config

def extract_layer_weights(state_dict, layer_idx=0):
    """提取指定层的权重"""
    layer_weights = {}
    prefix = f"layers.{layer_idx}."
    
    for key, value in state_dict.items():
        if key.startswith(prefix):
            # 移除层前缀，保留相对路径
            relative_key = key[len(prefix):]
            layer_weights[relative_key] = value
    
    return layer_weights

def test_transformer_block_comparison():
    """对比官方实现和自定义实现"""
    print("=== Transformer Block 对比测试 ===")
    
    # 加载测试数据
    state_dict, config = load_test_data()
    
    # 提取配置参数
    d_model = config['d_model']
    num_heads = config['num_heads']
    d_ff = config['d_ff']
    max_seq_len = config['context_length']
    theta = config['rope_theta']
    
    print(f"配置参数:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  theta: {theta}")
    
    # 创建测试输入
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 8  # 使用较短的序列长度
    in_features = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n输入形状: {in_features.shape}")
    
    # 提取第0层的权重
    layer_weights = extract_layer_weights(state_dict, layer_idx=0)
    
    print(f"\n提取的权重键:")
    for key in sorted(layer_weights.keys()):
        print(f"  {key}: {layer_weights[key].shape}")
    
    # 测试官方实现
    print(f"\n--- 官方实现测试 ---")
    try:
        # 构建完整的权重字典（官方adapter需要的格式）
        full_weights = {}
        for key, value in layer_weights.items():
            full_weights[key] = value
        
        official_output = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            theta=theta,
            weights=full_weights,
            in_features=in_features
        )
        
        print(f"✅ 官方实现成功")
        print(f"输出形状: {official_output.shape}")
        print(f"输出范围: [{official_output.min().item():.4f}, {official_output.max().item():.4f}]")
        print(f"输出均值: {official_output.mean().item():.4f}")
        print(f"输出标准差: {official_output.std().item():.4f}")
        
    except Exception as e:
        print(f"❌ 官方实现失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试自定义实现
    print(f"\n--- 自定义实现测试 ---")
    try:
        # 修正权重键名以匹配自定义实现的期望格式
        custom_weights = {}
        
        # 注意力权重映射
        if 'attn.q_proj.weight' in layer_weights:
            custom_weights['attn.q_proj.weight'] = layer_weights['attn.q_proj.weight']
        if 'attn.k_proj.weight' in layer_weights:
            custom_weights['attn.k_proj.weight'] = layer_weights['attn.k_proj.weight']
        if 'attn.v_proj.weight' in layer_weights:
            custom_weights['attn.v_proj.weight'] = layer_weights['attn.v_proj.weight']
        if 'attn.output_proj.weight' in layer_weights:
            custom_weights['attn.output_proj.weight'] = layer_weights['attn.output_proj.weight']
        
        # FFN权重映射
        if 'ffn.w1.weight' in layer_weights:
            custom_weights['ffn.w1.weight'] = layer_weights['ffn.w1.weight']
        if 'ffn.w2.weight' in layer_weights:
            custom_weights['ffn.w2.weight'] = layer_weights['ffn.w2.weight']
        if 'ffn.w3.weight' in layer_weights:
            custom_weights['ffn.w3.weight'] = layer_weights['ffn.w3.weight']
        
        # 归一化权重映射
        if 'ln1.weight' in layer_weights:
            custom_weights['ln1.weight'] = layer_weights['ln1.weight']
        if 'ln2.weight' in layer_weights:
            custom_weights['ln2.weight'] = layer_weights['ln2.weight']
        
        print(f"自定义权重键:")
        for key in sorted(custom_weights.keys()):
            print(f"  {key}: {custom_weights[key].shape}")
        
        custom_transformer = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            theta=theta,
            weights_dict=custom_weights
        )
        
        custom_output = custom_transformer(in_features)
        
        print(f"✅ 自定义实现成功")
        print(f"输出形状: {custom_output.shape}")
        print(f"输出范围: [{custom_output.min().item():.4f}, {custom_output.max().item():.4f}]")
        print(f"输出均值: {custom_output.mean().item():.4f}")
        print(f"输出标准差: {custom_output.std().item():.4f}")
        
    except Exception as e:
        print(f"❌ 自定义实现失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 对比结果
    print(f"\n--- 结果对比 ---")
    
    # 形状对比
    shapes_match = official_output.shape == custom_output.shape
    print(f"形状匹配: {shapes_match}")
    
    if shapes_match:
        # 数值对比
        max_diff = torch.max(torch.abs(official_output - custom_output)).item()
        mean_diff = torch.mean(torch.abs(official_output - custom_output)).item()
        relative_diff = mean_diff / (torch.mean(torch.abs(official_output)).item() + 1e-8)
        
        print(f"最大绝对差异: {max_diff:.6f}")
        print(f"平均绝对差异: {mean_diff:.6f}")
        print(f"相对差异: {relative_diff:.6f}")
        
        # 判断是否接近
        tolerance = 1e-4
        is_close = torch.allclose(official_output, custom_output, atol=tolerance, rtol=tolerance)
        print(f"数值接近 (tolerance={tolerance}): {is_close}")
        
        if not is_close:
            print(f"\n详细差异分析:")
            diff = torch.abs(official_output - custom_output)
            print(f"差异分布:")
            print(f"  最小差异: {diff.min().item():.6f}")
            print(f"  25%分位数: {torch.quantile(diff, 0.25).item():.6f}")
            print(f"  中位数: {torch.median(diff).item():.6f}")
            print(f"  75%分位数: {torch.quantile(diff, 0.75).item():.6f}")
            print(f"  最大差异: {diff.max().item():.6f}")
            
            # 找出差异最大的位置
            max_diff_idx = torch.argmax(diff)
            max_diff_pos = torch.unravel_index(max_diff_idx, diff.shape)
            print(f"最大差异位置: {max_diff_pos}")
            print(f"  官方值: {official_output[max_diff_pos].item():.6f}")
            print(f"  自定义值: {custom_output[max_diff_pos].item():.6f}")
    
    else:
        print(f"❌ 输出形状不匹配，无法进行数值对比")
        print(f"  官方输出形状: {official_output.shape}")
        print(f"  自定义输出形状: {custom_output.shape}")

def test_without_weights():
    """测试无权重情况下的实现"""
    print(f"\n=== 无权重测试 ===")
    
    # 基本参数
    d_model = 64
    num_heads = 4
    d_ff = 128
    max_seq_len = 16
    theta = 10000.0
    
    # 创建测试输入
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 8
    in_features = torch.randn(batch_size, seq_len, d_model)
    
    try:
        # 测试自定义实现（无权重）
        custom_transformer = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            theta=theta,
            weights_dict=None
        )
        
        output = custom_transformer(in_features)
        
        print(f"✅ 无权重测试成功")
        print(f"输入形状: {in_features.shape}")
        print(f"输出形状: {output.shape}")
        print(f"形状匹配: {output.shape == in_features.shape}")
        
        # 检查梯度流
        loss = output.sum()
        loss.backward()
        
        has_gradients = any(p.grad is not None for p in custom_transformer.parameters())
        print(f"梯度计算正常: {has_gradients}")
        
    except Exception as e:
        print(f"❌ 无权重测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_transformer_block_comparison()
    test_without_weights()
    print(f"\n🎉 测试完成！")