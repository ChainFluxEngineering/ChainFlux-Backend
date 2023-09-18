from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
import os
import openai
from PyPDF2 import PdfReader
import numpy as np
from langchain.llms import OpenAI

from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
import tempfile
import hashlib
from langchain.chains.question_answering import load_qa_chain

os.environ["OPENAI_API_KEY"] = "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"
app = Flask(__name__)
CORS(app)
s3_client = boto3.client('s3')
s3_bucket = 'janak-mvp'

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    try:
        
        # Get the uploaded file from the request
        pdf_file = request.files['pdf']
        s3_folder = request.form['folder']
        user_id = request.form['user_id']
       
        
        # print(text)



        # Check if the file is empty
        if not pdf_file or not s3_folder:
            return jsonify({'error': 'File and folder name are required'}), 400
        # Create a temporary file to store the content of the PDF
        temp_pdf_file = tempfile.NamedTemporaryFile(delete=False)
        pdf_content = pdf_file.read()
        temp_pdf_file.write(pdf_content)
        temp_pdf_file.close()

        # Specify the S3 folder and file name
        s3_file = pdf_file.filename

        # Upload the temporary file to S3
        s3_client.upload_file(temp_pdf_file.name, s3_bucket, os.path.join(s3_folder, s3_file))

        # Read the PDF content from the temporary file and split it into chunks
        text = ""
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text+= page.extract_text()

        # print(text)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function = len)
        docs = text_splitter.split_text(text)
        
        # Create embeddings from the chunks
        # embeddings = OpenAIEmbeddings(model_name="ada")
        # for chunks in docs:
        #     query_result = embeddings.embed_query(chunks)
          
        # query_result = embeddings.embed_query("Hello world")
        # l=len(query_result)

        # # Store the embeddings to Pinecone
        pinecone.init( 
            api_key='aa654506-71ba-4d19-a1b8-56dc329b7090',
            environment='us-east-1-aws'
        )
        index_name = user_id
        index_exists = False

        indexes = pinecone.list_indexes()
      
        if index_name in indexes:
            index_exists = True


        if not index_exists:
        # Create the index if it doesn't exist
         pinecone.create_index(index_name, dimension=1536, metric='cosine')
        

        index = pinecone.Index(index_name)
    
        embeddings = OpenAIEmbeddings(model_name="ada")
        index = Pinecone.from_texts(docs, embeddings, index_name=index_name)
        
    

        response_data = {'message': f'Index {index_name} created and documents uploaded successfully', 'embedding_length': len(docs)}

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/query', methods=['POST'])
def query():
    try:
        
        
        query = request.form['query']
        user_id = request.form['user_id']
       
        
       
       


        text = "Hello this is test insert"
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function = len)
        docs = text_splitter.split_text(text)
        

   
        pinecone.init( 
            api_key='aa654506-71ba-4d19-a1b8-56dc329b7090',
            environment='us-east-1-aws'
        )
        index_name = user_id
        index_exists = False
        print("ok1")
        indexes = pinecone.list_indexes()
         
        if index_name in indexes:
            index_exists = True

        if not index_exists:
         return jsonify({'error': f'Index {index_name} does not exist.'}), 404
        print("ok")
        embeddings = OpenAIEmbeddings(model_name="ada")
        embedding = embeddings.embed_query(docs[0])
        index = pinecone.Index(index_name)
        index = Pinecone.from_texts(docs, embeddings, index_name=index_name)
        
        print(index)
        k=3 
        score = False
        if score:
          similar_docs = index.similarity_search_with_score(query,k=k)
        else:
         similar_docs = index.similarity_search(query,k=k)
        similar_docs
        
        # model_name = "text-davinci-003"
        model_name = "gpt-3.5-turbo"
        # model_name = "gpt-4"
        llm = OpenAI(model_name=model_name)


        chain = load_qa_chain(llm, chain_type="stuff")

        answer =  chain.run(input_documents=similar_docs, question=query)
        
        

        response_data = {'message': 'query successfully', 'answer': answer}

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)