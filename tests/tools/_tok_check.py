from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/public/share/GLM-5-W4A8_V2_6", trust_remote_code=True)
p = "The capital of France is"
ids = tok(p).input_ids
print("call:", ids, len(ids))
print("encode:", tok.encode(p), len(tok.encode(p)))
print("no_special:", tok(p, add_special_tokens=False).input_ids)
print("bos:", tok.bos_token_id, tok.bos_token)
print("tokens:", tok.convert_ids_to_tokens(ids))
