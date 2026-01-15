import numpy as np

"""
LAR@k LLM Agreement Rate 
"""
# # 增加凌云API进行GPT-5 check
def compute_lar(mllm_yes_no):
    # mllm_yes_no: shape (N,), values in {0,1}
    return mllm_yes_no.mean()


def auc_k(lar_dict, k_list):
    ys = [lar_dict[k] for k in k_list]
    return np.trapz(ys, k_list) / (max(k_list)-min(k_list))


def compute_k_star(lar_dict, rho=0.8):
    for k in sorted(lar_dict):
        if lar_dict[k] >= rho:
            return k
    return max(lar_dict)+1



import base64
from openai import OpenAI

# =========================
# OpenAI client（yunai.chat）
# =========================

client = OpenAI(
    base_url="url",
    api_key="SecriteKey"   # ← 替换成你的 key
)


MODEL_NAME = "gpt-5.2"
# gpt-4o
# gpt-5.2

# =========================
# 图像编码（base64）
# =========================
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =========================
# Prompt 构造（MLLM faithfulness）
# =========================
def build_prompt(pred_label, concepts):
    concept_text = "\n".join([f"- {c}" for c in concepts])

    prompt = f"""
        You are a medical expert reviewing an AI model's diagnostic explanation.

        You are given:
        (1) a medical image,
        (2) a list of diagnostic concepts extracted by the model, and
        (3) the model's predicted diagnosis.

        Predicted diagnosis: {pred_label}

        Diagnostic concepts:
        {concept_text}

        Your task is NOT to determine whether the diagnosis is correct.
        Instead, assess whether the provided concepts are sufficient to justify the predicted diagnosis based on the image.

        Do the listed concepts, taken together, provide adequate evidence to support the predicted diagnosis?

        Answer with a single word: Yes or No
    """
    return prompt.strip()


# =========================
# 核心：MLLM_Check（base64 版本）
# =========================
def MLLM_Check(image_path, pred_label, concepts):
    """
    使用 base64 + OpenAI SDK 的 MLLM faithfulness check
    return: "Yes" or "No"
    """

    base64_image = encode_image(image_path)
    prompt = build_prompt(pred_label, concepts)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,      # 保证确定性
        max_tokens=5,         # 只输出 Yes / No
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )
    

    answer = response.choices[0].message.content.strip()
    print(answer)

    if answer not in ["Yes", "No"]:
        raise ValueError(f"Unexpected MLLM output: {answer}")

    return answer


# =========================
# ISIC-2018 测试
# =========================
if __name__ == "__main__":

    image_path = "/isic2018/test/a vascular lesion/test_6909.png"
    pred_label = "Melanoma"

    concepts = [
        "highly variable, often with multiple colors",
        "irregular borders",
        "asymmetrical structure"
    ]

    result = MLLM_Check(
        image_path=image_path,
        pred_label=pred_label,
        concepts=concepts
    )

    print("MLLM judgment:", result)

