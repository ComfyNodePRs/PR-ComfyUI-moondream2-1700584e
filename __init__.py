import huggingface_hub, folder_paths
from .moondream import onnx_vl
import torchvision.transforms
import pathlib, gzip

ROOT = pathlib.Path(folder_paths.models_dir) / "LLM"

class moondream2_DownLoad:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
            "size": (
                ["0.5b", "2b"],
                { "default": "2b" }),

            "quant": (
                ["int4", "int8"],
                { "default": "int8" })
        } }

    RETURN_TYPES = ("moondream2", )
    RETURN_NAMES = ("model", )

    FUNCTION = "load"
    CATEGORY = "moondream2"

    def download(self, name, dest):
        # download gzipped model
        gz = huggingface_hub.hf_hub_download(
            repo_id="vikhyatk/moondream2",
            filename=f"{name}.gz", revision="onnx")

        dest.parent.mkdir(exist_ok=True, parents=True)

        with gzip.open(gz) as data:
            read_chk = lambda: data.read(1024576)
            with open(dest, "wb") as mf:
                for chk in iter(read_chk, b""): mf.write(chk)

        # delete the gzip from hf cache
        url = huggingface_hub.hf_hub_url(
            repo_id="vikhyatk/moondream2",
            filename=f"{name}.gz", revision="onnx")

        hash_ = huggingface_hub \
            .get_hf_file_metadata(url).commit_hash

        cache = huggingface_hub.scan_cache_dir()
        cache.delete_revisions(hash_).execute()

    def load(self, size, quant):
        size = size.replace(".", "_")
        name = f"moondream-{size}-{quant}.mf"

        mf = ROOT / name
        if not mf.exists(): self.download(name, mf)

        model = onnx_vl.OnnxVL.from_path(mf.as_posix())
        return (model, )

class moondream2_Encode:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
            "model": ("moondream2", ),
            "image": ("IMAGE", ),
        } }

    RETURN_TYPES = ("MD2_IMAGE", )
    RETURN_NAMES = ("md2_image", )

    FUNCTION = "encode"
    CATEGORY = "moondream2"

    def encode(self, model, image):
        img = image[0].permute(2, 0, 1)
        img = torchvision.transforms \
            .v2.functional.to_pil_image(img)

        return (model.encode_image(img), )

class moondream2_Caption:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
            "model": ("moondream2", ),
            "md2_image": ("MD2_IMAGE", ),

            "length": (
                ["short", "long"],
                { "default": "short" }),

            "max_tokens": ("INT", { "default": 512 })
        } }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("caption", )

    FUNCTION = "caption"
    CATEGORY = "moondream2"

    def caption(self, model, md2_image, length, max_tokens):
        if length == "long": length = "normal"

        result = model.caption(
            md2_image, length, stream=False,
            settings={ "max_tokens": max_tokens })

        return (result["caption"], )

class moondream2_Query:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
            "model": ("moondream2", ),
            "md2_image": ("MD2_IMAGE", ),

            "question": ("STRING", { "default": "What is this?" }),
            "max_tokens": ("INT", { "default": 512 })
        } }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("answer", )

    FUNCTION = "caption"
    CATEGORY = "moondream2"

    def caption(self, model, md2_image, question, max_tokens):
        result = model.query(
            md2_image, question, stream=False,
            settings={ "max_tokens": max_tokens })

        return (result["answer"], )

NODE_CLASS_MAPPINGS = {
    "moondream2_DownLoad": moondream2_DownLoad,
    "moondream2_Encode": moondream2_Encode,
    "moondream2_Caption": moondream2_Caption,
    "moondream2_Query": moondream2_Query
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "moondream2_DownLoad": "moondream2 (Down)Load",
    "moondream2_Encode": "moondream2 Encode Image",
    "moondream2_Caption": "moondream2 Caption",
    "moondream2_Query": "moondream2 Query"
}
