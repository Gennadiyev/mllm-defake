from mllm_defake.utils import encode_image_to_base64


def v1(cache: dict) -> None:
    cache["few_shot_image_url_0"] = encode_image_to_base64(r"WildFakeResampled\real\3739320002682.jpg")
    cache[
        "few_shot_result_0"
    ] = """The image showcases a beautifully restored classic GMC pickup truck, likely from the 1960s. The vehicle features a deep maroon paint job, a customized lowered stance, and large chrome wheels. The front grille and bumper are polished to a mirror-like shine, reflecting the surroundings. The hood has a subtle raised center section, adding to the aggressive yet classic aesthetic. The background consists of leafless trees, a concrete sidewalk, and some parked cars, indicating a late autumn or winter setting.

    1. **Real-World Imperfections** – This photograph contains natural reflections, minor inconsistencies in lighting, and real-world textures that AI-generated images often struggle to replicate perfectly. AI images usually have overly smooth or exaggerated reflections that may not align naturally.

    2. **Environmental Realism** – The shadows, reflections, and slight distortions in the chrome and glass suggest a real camera lens was used. AI-generated images often lack physically accurate light interactions or introduce unrealistic shadowing.

    3. **Fine Detail Accuracy** – Elements like the chrome reflections, minor imperfections in paint, and dust or debris on the pavement are natural in real images but are sometimes exaggerated, missing, or blurred in AI-generated images.

    4. **Lens Effects** – This image exhibits subtle lens characteristics such as depth of field and focal point sharpness, which AI models sometimes mimic but often with inconsistencies (e.g., unnatural depth blurring).

    5. **Background Integration** – The truck integrates naturally with its environment, with proper perspective and lighting. AI-generated images sometimes have odd blending issues where objects appear out of place or have mismatched lighting.

    Based on these observations, the image is likely a photograph of a real vehicle rather than a computer-generated rendering."""
    cache["few_shot_image_url_1"] = encode_image_to_base64(
        r"WildFakeResampled\fake\2f5aa45674071990686e41a67e5889bc.png"
    )
    cache[
        "few_shot_result_1"
    ] = """The image attempts to depict a desk setup. However, the texture of the chair is asymmetrical as it only has two legs; meanwhile, the computer on the desk is skewed. The shadow is also poorly rendered. The wooden drawer on the right side of the desk have a wrong perspective. The blue coat on the desk is meaningless and does not fit the scene, a typical error in AI-generated images.

    Therefore, the image is likely computer-generated due to the unrealistic elements and inconsistencies in the scene."""
