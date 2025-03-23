import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import os

# Load LLaMA Model Path from Environment Variable (Optional)
MODEL_PATH = os.getenv("MODEL_PATH", "model/llama-2-7b-chat.ggmlv3.q8_0.bin")

## Function To Get Response from LLaMA 2 Model
def getLLamaresponse(input_text, no_words, blog_style):
    """Generate blog content using LLaMA 2 model."""
    
    # Ensure the model file exists
    if not os.path.exists(MODEL_PATH):
        return "Error: LLaMA model file not found. Please check the path."

    # Load LLaMA Model
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        config={"max_new_tokens": 256, "temperature": 0.01},
    )

    # Improved Prompt Template for structured output
    template = """
    You are a professional blog writer. Write a structured blog on "{input_text}" for a {blog_style} audience.

    The blog should include:
    1. **Title**
    2. **Introduction**
    3. **Research-based Content**
    4. **Key Points**
    5. **Conclusion**

    Word Limit: {no_words} words.
    """
    
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template,
    )

    # Generate Response from the LLaMA 2 Model
    formatted_prompt = prompt.format_prompt(
        blog_style=blog_style, input_text=input_text, no_words=no_words
    ).to_string()

    response = llm(formatted_prompt)
    return response

# Streamlit UI
st.set_page_config(page_title="AI Blog Generator", page_icon="🤖", layout="centered")

st.header("Generate Blogs with LLaMA 2 🤖")

# User Inputs
input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns(2)
with col1:
    no_words = st.text_input("Number of Words")
with col2:
    blog_style = st.selectbox("Writing the blog for", ("Researchers", "Data Scientist", "Common People"), index=0)

submit = st.button("Generate")

if submit:
    if input_text and no_words:
        with st.spinner("Generating blog... ⏳"):
            blog = getLLamaresponse(input_text, no_words, blog_style)
        
        # Splitting the response into sections
        sections = blog.split("\n")  # Assuming response has newlines

        st.subheader("Generated Blog:")
        for section in sections:
            if "**Introduction:**" in section:
                st.markdown("### 📝 Introduction")
            elif "**Research-based Content:**" in section:
                st.markdown("### 📚 Research-based Content")
            elif "**Key Points:**" in section:
                st.markdown("### 🔑 Key Points")
            elif "**Conclusion:**" in section:
                st.markdown("### 🏁 Conclusion")
            
            st.write(section)  # Display the content

    else:
        st.warning("⚠️ Please enter a topic and number of words!")