from openai import OpenAI
import base64

def ask_online(image_data, prompt, model_choice = "claude-3-5-sonnet-20240620"):
    client = OpenAI(
        api_key="sk-DTEqVoi3wnhN5IIYCxB6dS0BS3mqmUGO1OsPO0v3HbpsFSEF",
        base_url="https://api.claudeshop.top/v1"
    )
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    response = client.chat.completions.create(
        model=model_choice,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            }
        ],
        max_tokens=300,
        stream=False  # # 是否开启流式输出
    )
    print(response.choices[0].message)
