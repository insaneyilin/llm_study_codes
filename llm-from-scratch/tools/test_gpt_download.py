from gpt_download import download_and_load_gpt2

CHOOSE_MODEL_SIZE = "gpt2-small (124M)"

model_size = CHOOSE_MODEL_SIZE.split(" ")[-1].lstrip("(").rstrip(")")

settings, params = download_and_load_gpt2(model_size=model_size,
                                          models_dir="gpt2_model_ckpts")
print(f"Model size: {model_size}")
print(f"Settings: {settings}")
print(f"Params: {params}")

for choose_model_size in [
        "gpt2-medium (355M)", "gpt2-large (774M)", "gpt2-xl (1558M)"
]:
    model_size = choose_model_size.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size,
                                              models_dir="gpt2_model_ckpts")
    print(f"Model size: {model_size}")
    print(f"Settings: {settings}")
    print(f"Params: {params}")
