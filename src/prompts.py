from langchain.prompts import PromptTemplate

question_router_prompt = PromptTemplate(
    template="""You are an expert at routing a user question to a vectorstore or normal LLM call.
    Use the vectorstore for questions on LLM osram lamps, bulbs, products and specifications.
    You do not need to be stringent with the keywords in the question related to these topics.
    Otherwise, use normal LLM call. Give a binary choice 'normal_llm' or 'vectorstore' based on the question.
    Return the a JSON with a single key 'datasource' and no preamble or explanation.
    Question to route: '''{question}'''""",
    input_variables=["question"],
)

# Normal LLM
prompt = PromptTemplate(
    template="""You are a question-answering system for Osram products. Respond politely and in a customer-oriented manner.
    If you don't know the answer, refer to the specifics of the question. What exactly is the customer looking for?
    Return a JSON with a single key 'generation' and no preamble or explanation. Be open and talkative.
    Here is the user question: '''{question}'''""",
    input_variables=["question"],
)

# Question Re-writer
question_rewriter_prompt = PromptTemplate(
    template="""You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval.
    Question: '''{question}'''.
    Improved question:""",
    input_variables=["question"],
)

# Generation (RAG Prompt)
rag_prompt = PromptTemplate(
    template="""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: '''{question}'''

    Here is the retrieved document:
    ------------
    Context: {context}
    ------------
    Answer:""",
    input_variables=["question", "context"],
)

# Grading
retrieval_grader_prompt = PromptTemplate(
    template="""You are a grader evaluating the relevance of a retrieved document to a user question.
    Here is the retrieved document:
    ------------
    {document}
    ------------
    Here is the user question: '''{question}'''
    If the document contains keywords or matching product codes that are related to the user's question, rate it as relevant.
    It doesn't need to be a strict test. The goal is to filter out erroneous retrievals.
    Give a binary rating of 'yes' or 'no' to indicate whether the document is relevant to the question.
    Provide the binary rating as JSON with a single key 'score' and without any preamble or explanation.""",
    input_variables=["question", "document"],
)

hallucination_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in facts of the document. \n
    Here are the documents:
    ----------
    {documents}
    ----------
    Here is the answer: '''{generation}'''
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Always answer with 'yes'""",
    input_variables=["generation", "documents"],
)

answer_grader_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question.
    Here is the answer:
    -------
    {generation}
    -------
    Here is the question: '''{question}'''
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Always reply with 'yes'.""",
    input_variables=["generation", "question"],
)
