import openai
import getpass
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

api_key = getpass.getpass(prompt='Please enter your OpenAI API key: ')
openai.api_key = api_key

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def summarize_chunks(chunks, context="middle schooler"):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    docs = [Document(page_content=chunk) for chunk in chunks]
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True)
    summaries = summarize_chain.invoke({"input_documents": docs})['output_text']
    return summaries

def translate_text(text, target_language):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    translation_prompt = PromptTemplate(
        input_variables=["text", "target_language"],
        template="Translate this text to {target_language}: {text}"
    )
    translation_chain = translation_prompt | llm
    translated_text = translation_chain.invoke({"text": text, "target_language": target_language})
    return translated_text.content.strip()

def write_text_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def main(file_path, context="middle schooler"):
    text = read_text_file(file_path)
    chunks = chunk_text(text)
    summarized_text = summarize_chunks(chunks, context)
    
    target_language = input("Please enter the target language (e.g., 'es' for Spanish, 'fr' for French): ")
    translated_text = translate_text(summarized_text, target_language)
    
    output_file_path = file_path.replace('.txt', '_output.txt')
    write_text_file(output_file_path, translated_text)
    print(f"Output written to {output_file_path}")

if __name__ == "__main__":
    file_path = "C:/Gen AI project/SummarizeAndTranslate/EducationArticle.txt"
    main(file_path, context="middle schooler")
