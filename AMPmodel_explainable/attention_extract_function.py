import torch

def extract_attention_weights(model, test_dataloader):
    merged_test0 = []
    merged_test1 = []
    merged_test2 = []
    merged_test3 = []
    merged_test4 = []
    merged_test5 = []

    # 遍历测试数据集，进行推断
    for inputs, _ in test_dataloader:
        # 前向传播
        with torch.no_grad():
            # 第0层
            inputs_norm0 = model.transformer_encoder.layers[0].norm1(inputs)
            test0 = model.transformer_encoder.layers[0].self_attn(
                inputs_norm0, inputs_norm0, inputs_norm0,
                need_weights=True, average_attn_weights=True
            )
            layer0_outcome = model.transformer_encoder.layers[0](inputs)
            
            # 第1层
            inputs_norm1 = model.transformer_encoder.layers[1].norm1(layer0_outcome)
            test1 = model.transformer_encoder.layers[1].self_attn(
                inputs_norm1, inputs_norm1, inputs_norm1,
                need_weights=True, average_attn_weights=True
            )
            layer1_outcome = model.transformer_encoder.layers[1](layer0_outcome)
            
            # 第2层
            inputs_norm2 = model.transformer_encoder.layers[2].norm1(layer1_outcome)
            test2 = model.transformer_encoder.layers[2].self_attn(
                inputs_norm2, inputs_norm2, inputs_norm2,
                need_weights=True, average_attn_weights=True
            )
            layer2_outcome = model.transformer_encoder.layers[2](layer1_outcome)
            
            # 第3层
            inputs_norm3 = model.transformer_encoder.layers[3].norm1(layer2_outcome)
            test3 = model.transformer_encoder.layers[3].self_attn(
                inputs_norm3, inputs_norm3, inputs_norm3,
                need_weights=True, average_attn_weights=True
            )
            layer3_outcome = model.transformer_encoder.layers[3](layer2_outcome)
            
            # 第4层
            inputs_norm4 = model.transformer_encoder.layers[4].norm1(layer3_outcome)
            test4 = model.transformer_encoder.layers[4].self_attn(
                inputs_norm4, inputs_norm4, inputs_norm4,
                need_weights=True, average_attn_weights=True
            )
            layer4_outcome = model.transformer_encoder.layers[4](layer3_outcome)
            
            # 第5层
            inputs_norm5 = model.transformer_encoder.layers[5].norm1(layer4_outcome)
            test5 = model.transformer_encoder.layers[5].self_attn(
                inputs_norm5, inputs_norm5, inputs_norm5,
                need_weights=True, average_attn_weights=True
            )
            layer5_outcome = model.transformer_encoder.layers[5](layer4_outcome)

            # 提取注意力权重
            test_attn0 = test0[1]
            test_attn1 = test1[1]
            test_attn2 = test2[1]
            test_attn3 = test3[1]
            test_attn4 = test4[1]
            test_attn5 = test5[1]

            # 将每个元素的第二个 tuple 中的张量添加到 merged_test 中
            merged_test0.extend([tensor.cpu().detach().numpy() for tensor in test_attn0])
            merged_test1.extend([tensor.cpu().detach().numpy() for tensor in test_attn1])
            merged_test2.extend([tensor.cpu().detach().numpy() for tensor in test_attn2])
            merged_test3.extend([tensor.cpu().detach().numpy() for tensor in test_attn3])
            merged_test4.extend([tensor.cpu().detach().numpy() for tensor in test_attn4])
            merged_test5.extend([tensor.cpu().detach().numpy() for tensor in test_attn5])

    return merged_test0, merged_test1, merged_test2, merged_test3, merged_test4, merged_test5