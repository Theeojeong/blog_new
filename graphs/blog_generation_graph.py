from utils.google_utils import google_search
from chains.embedding_chain import update_embedding_cache
from chains.outline_chain import create_outline_with_additional_info
from chains.content_chain import generate_blog_content
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as LangchainOpenAI
from config import DB_CONFIG, OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID

def blog_generation_workflow(product_name, product_specs_list, blog_title, keywords):

    specs_info_list = []
    used_urls = []  # 사용한 URL을 저장할 리스트

    for spec in product_specs_list:
        query = f"{spec} 설명"
        search_results = google_search(query, num=3, google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)
        if not search_results:
            product_details_web = "웹 검색 결과를 찾지 못했습니다."
        else:
            urls = [item.get("link") for item in search_results[:5]]
            # urls를 used_urls에 추가
            used_urls.extend(urls)

            loader = UnstructuredURLLoader(urls=urls)
            docs = loader.load()

            if not docs:
                product_details_web = "웹 검색에서 추가 상세정보를 찾지 못했습니다."
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                split_docs = text_splitter.split_documents(docs)

                if len(split_docs) == 0:
                    product_details_web = "텍스트 청크를 생성하지 못했습니다. 문서 내용이 비어있는지 확인하세요."
                

                else:
                    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                    vectorstore = FAISS.from_documents(split_docs, embeddings)
                    retriever = vectorstore.as_retriever(search_type="similarity", search_k=3)

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=LangchainOpenAI(openai_api_key=OPENAI_API_KEY),
                        chain_type="stuff",
                        retriever=retriever
                    )
                    question = f"한국어로 {query} 상세 스펙에 대한 설명을 해줘"
                    answer = qa_chain({"query": question})
                    product_details_web = answer["result"]

                update_embedding_cache(spec, product_details_web)

        specs_info_list.append((spec, product_details_web, None))

    outline, combined_info = create_outline_with_additional_info(product_name, specs_info_list, blog_title, keywords)

    max_retries = 2
    generated_content = None
    for attempt in range(max_retries):
        generated_content = generate_blog_content(outline, blog_title, keywords, product_name, OPENAI_API_KEY)

    return generated_content, used_urls