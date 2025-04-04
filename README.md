# 📝 AI Blog Generator (LLaMA 2 + Streamlit)

🚀 **AI-powered blog generator** that creates **structured, research-backed blog articles** using **LLaMA 2** and **Streamlit**. Supports **Wikipedia & Google Search research**, **custom word limits**, and **automated section formatting** (Introduction, Key Points, Conclusion).  

---

![Blog Screenshot](assets/testimonal.png)


## 🎯 **Features**
✅ **AI-powered blog generation** using LLaMA 2  
✅ **Research-based content** (Wikipedia & Google Search integration)  
✅ **Structured blog format** (Title, Introduction, Key Points, Conclusion)  
✅ **Customizable word count & target audience**  
✅ **Simple, interactive Streamlit UI**  

---

## 🛠️ **Installation & Setup**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/sreevamsi2005/ai-blog-generator.git

cd ai-blog-generator
```
2️⃣ Install Dependencies
  ```bash
pip install -r requirements.txt
```
3️⃣ Set Up LLaMA 2 Model
## 📥 Download LLaMA 2 Model and use locally ,
You can download the **LLaMA 2 (7B Chat GGML)** model from [Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main).

Ensure your LLaMA 2 model file (llama-2-7b-chat.ggmlv3.q8_0.bin) is in the correct directory:
```bash
models/llama-2-7b-chat.ggmlv3.q8_0.bin
```
4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```
## 🎮 Usage
1️⃣ **Enter a blog topic** in the input field  
2️⃣ **Select a target audience** (Researchers, Data Scientists, Common People)  
3️⃣ **Choose the word limit**  
4️⃣ **Click "Generate"** to create a structured blog  
5️⃣ **View AI-generated blogs** with **formatted sections**  

