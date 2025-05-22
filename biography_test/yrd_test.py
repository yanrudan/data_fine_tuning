from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
        "/home/tangzhenheng/models/gpt2/gpt2-small",
        model_max_length=512,  # unimportant config
        padding_side="right",
        use_fast=True,
        )
text = "Today carries the quiet weight of autumn’s first chill—a mingling of nostalgia and renewal. The sunlight, pale yet persistent, mirrors my own tempered optimism. There’s beauty in this transient balance between what was and what might be."

print(tokenizer.__class__.__name__)
prefix_token_list = tokenizer(text)["input_ids"]
print(prefix_token_list)