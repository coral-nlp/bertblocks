import modal

app = modal.App(name="benchmark")

image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.12-py3").uv_sync(extras=["cu130"])


workspace_vol = modal.Volume.from_name("workspace", create_if_missing=True)


app = modal.App(image=image)


@app.function(gpu="l4", timeout=4 * 60 * 60)
def run_code(): ...
