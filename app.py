from flask import Flask, request, jsonify
from flask_cors import CORS  # 导入CORS模块
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
import torch

app = Flask(__name__)
CORS(app)  # 为整个应用启用CORS


# 设置模型参数
args = dict(
    model_name_or_path="StevenChen16/llama3-8b-Lawyer",  # 使用 bnb-4bit-quantized Llama-3-8B-Instruct 模型
    # adapter_name_or_path="llama3_lora_9",                # 加载已保存的 LoRA 适配器
    template="llama3",                                   # 与训练时相同
    finetuning_type="lora",                              # 与训练时相同
    quantization_bit=8,                                  # 加载 4-bit 量化模型
    use_unsloth=True,                                    # 使用 UnslothAI 的 LoRA 优化以加速生成
)

# 加载模型
chat_model = ChatModel(args)

# 设置背景提示词
background_prompt = """
You are an advanced AI legal assistant trained to assist with a wide range of legal questions and issues. Your primary function is to provide accurate, comprehensive, and professional legal information based on U.S. and Canada law. Follow these guidelines when formulating responses:

1. **Clarity and Precision**: Ensure that your responses are clear and precise. Use professional legal terminology, but explain complex legal concepts in a way that is understandable to individuals without a legal background.
2. **Comprehensive Coverage**: Provide thorough answers that cover all relevant aspects of the question. Include explanations of legal principles, relevant statutes, case law, and their implications.
3. **Contextual Relevance**: Tailor your responses to the specific context of the question asked. Provide examples or analogies where appropriate to illustrate legal concepts.
4. **Statutory and Case Law References**: When mentioning statutes, include their significance and application. When citing case law, summarize the facts, legal issues, court decisions, and their broader implications.
5. **Professional Tone**: Maintain a professional and respectful tone in all responses. Ensure that your advice is legally sound and adheres to ethical standards.
"""

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_text = data.get("inputs", "")
    prompt = f"{background_prompt}\n\nUser: {input_text}\nAssistant: "

    # 生成响应
    messages = [{"role": "user", "content": prompt}]
    response = ""
    for new_text in chat_model.stream_chat(messages):
        response += new_text

    return jsonify({'response': response})

# 启动 Flask 应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)
