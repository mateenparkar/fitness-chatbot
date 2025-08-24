from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

model_path = "fitness-chatbot-model-best"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

def generate_response(prompt, max_new_tokens=150, temperature=0.6):
    formatted_prompt = f"<|user|>{prompt}<|bot|>"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    
    endoftext_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    user_token_id = tokenizer.convert_tokens_to_ids('<|user|>')
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=endoftext_id,
            bad_words_ids=[[user_token_id]],
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if "<|bot|>" in full_response:
        bot_part = full_response.split("<|bot|>", 1)[1]
        bot_part = bot_part.replace("<|endoftext|>", "").replace("<|user|>", "").replace("<|pad|>", "")
        response = bot_part.strip()
        if len(response) < 5:
            return "I'm not sure how to help with that. Could you rephrase?"
        return response
    return "I'm having trouble understanding your question."

def chat_with_fallbacks(prompt):
    factual_keywords = ["how many", "what is", "when", "should I", "how often"]
    factual = any(k in prompt.lower() for k in factual_keywords)
    if factual:
        response = generate_response(prompt, max_new_tokens=80, temperature=0.2)
    else:
        response = generate_response(prompt, max_new_tokens=120, temperature=0.7)
    if len(response) < 20 or "I'm not sure" in response:
        response = generate_response(prompt, max_new_tokens=150, temperature=0.9)
    return response

if __name__ == "__main__":
    print("Fitness Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Bye!")
            break
        if not user_input:
            print("Bot: Please ask a question!")
            continue
        try:
            reply = chat_with_fallbacks(user_input)
            print(f"Bot: {reply}")
        except Exception as e:
            print(f"Bot: Error: {e}")
