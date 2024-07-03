# Summarize and Translate 

This repository contains a Python script that leverages OpenAI's GPT-3.5-turbo model and the LangChain library to summarize and translate large text files. The script reads an input text file, breaks the content into manageable chunks, summarizes the text, translates the summary into a specified language, and saves the translated summary to a new text file.

Features Chunking: The text is split into overlapping chunks to preserve context using LangChain's CharacterTextSplitter.

Summarization: Each chunk is summarized independently using the Map-Reduce approach with OpenAI's GPT-3.5-turbo model.

Translation: The combined summary is translated into a specified target language using OpenAI's GPT-3.5-turbo model.

Output: The final translated summary is saved to a new text file in the same directory as the input file.

Requirements Python 3.7 or higher

OpenAI API key

Required Python packages:

openai

langchain

langchain-openai
