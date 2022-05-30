def deal_with_layers(prefix, layer_num, last):
    ans = []
    for i in range(layer_num):
        ans.append(prefix + str(i) + last)
    return ans


def analysis_quantize(name, model_type='bert'):

    if name is None:
        return None
    name_except = {
        'weight_embedding': [
            model_type + '.embeddings.word_embeddings.weight_fake_quant',
            model_type + '.embeddings.position_embeddings.weight_fake_quant',
            model_type + '.embeddings.token_type_embeddings.weight_fake_quant',
        ],
        'embedding_output': [
            model_type + '.embeddings.dropout_post_act_fake_quantize'
        ],
        'query_output': deal_with_layers(
            model_type + '.encoder.layer.', 12,
            '.attention.self.query_permute_post_act_fake_quantize'
            ),
        'key_output': deal_with_layers(
            model_type + '.encoder.layer.', 12,
            '.attention.self.key_transpose_post_act_fake_quantize'
            ),
        'value_output': deal_with_layers(
            model_type + '.encoder.layer.', 12,
            '.attention.self.value_permute_post_act_fake_quantize'
            ),
        'attention_probs_output': deal_with_layers(
            model_type + '.encoder.layer.', 12,
            '.attention.self.attention_probs_post_act_fake_quantize'
            ),
        'context_output': deal_with_layers(
            model_type + '.encoder.layer.', 12,
            '.attention.self.context_view_post_act_fake_quantize'
            ),
        'selfoutput_output': deal_with_layers(
            model_type + '.encoder.layer.', 12,
            '.attention.output.dropout_post_act_fake_quantize',
        ),
        'attention_output': deal_with_layers(
            model_type + '.encoder.layer.', 12,
            '.attention.output.layernorm_post_act_fake_quantize',
            ),
        'intermediate_output': deal_with_layers(
            model_type + '.encoder.layer.', 12,
            '.intermediate.intermediate_act_fn_post_act_fake_quantize'
            ),
        'output_output': deal_with_layers(
            model_type + '.encoder.layer.', 12,
            '.output.dropout_post_act_fake_quantize'
        ),
        'ffn_output': deal_with_layers(
            model_type + '.encoder.layer.', 11,
            '.output.layernorm_post_act_fake_quantize'
            ),
        'cls_item_output': ['bert.pooler.getitem_post_act_fake_quantize'],
        'pooler_output': ['dropout_post_act_fake_quantize']
    }
    if model_type == 'roberta':
        name_except['cls_item_output'] = ['classifier.getitem_post_act_fake_quantize']
        name_except['pooler_output'] = ['classifier.dropout_post_act_fake_quantize']
    return name_except[name]
