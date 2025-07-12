from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def create_prompt(vector_store):
    """
    Creates a prompt template for the retrieval chain.
    """
    retriever = vector_store.as_retriever()

    system_prompt = '''
    Use o contexto para responder as perguntas.
    Contexto: {context}
    '''

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            ('human', '{input}'),
        ]
    )

    return prompt, retriever

def create_chain(prompt, retriever, OPENAI_MODEL):
    """
    Creates a retrieval chain using the vector store.
    """
    model = ChatOpenAI( model=OPENAI_MODEL, temperature=0.7, max_tokens=1000)

    question_answer_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
    )

    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )

    return chain
