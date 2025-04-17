from transformers import RoFormerConfig, MPNetConfig, MPNetModel
from model import NewModel, NewConfig, AsmEncoder, AsmEncoderConfig

asm_config = AsmEncoderConfig(
    hidden_size=32,
    vocab_size=33555,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=128,
    max_position_embeddings=1024,
    use_bias=False,
)
asm_model = AsmEncoder(asm_config)

desc_config = MPNetConfig(
    hidden_size=32,
    vocab_size=30527,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=128,
    max_position_embeddings=1024,
)
desc_model = MPNetModel(desc_config)

src_config = NewConfig(
    hidden_size=32,
    vocab_size=33555,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=128,
    max_position_embeddings=1024,
)
src_model = NewModel(src_config)

asm_model.save_pretrained("models/example_asm", safe_serialization=False)
desc_model.save_pretrained("models/example_desc", safe_serialization=False)
src_model.save_pretrained("models/example_src", safe_serialization=False)
    